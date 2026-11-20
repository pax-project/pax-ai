//! A paper's full text, extracted from its PDF with poppler's `pdftotext`,
//! or, for a scanned PDF without a usable text layer, read by OCR: poppler's
//! `pdftoppm` renders each page and `tesseract` reads it.
//!
//! The references section is stripped: it's a list of other papers' titles,
//! which would make every paper look similar to the ones it cites.

use std::ffi::OsStr;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Part of every full-text store key: bump it when a change here alters the
/// text extracted from the same PDF, so old entries read as misses.
pub const EXTRACTOR_VERSION: u32 = 1;

/// The extractor's binary, looked up on `PATH` (the flake's `pax-mcp` wraps
/// poppler onto it).
const PDFTOTEXT: &str = "pdftotext";

/// The OCR path's binaries, also looked up on `PATH`: poppler's `pdfinfo`
/// counts the pages and `pdftoppm` renders them; `tesseract` reads them.
const PDFINFO: &str = "pdfinfo";
const PDFTOPPM: &str = "pdftoppm";
const TESSERACT: &str = "tesseract";

/// The resolution pages are rendered at for OCR, tesseract's sweet spot.
const OCR_DPI: &str = "300";

/// A text layer averaging fewer non-whitespace characters per page than
/// this is taken to be a scan with, at most, a typed cover page.
const MIN_CHARS_PER_PAGE: usize = 100;

/// A references heading before this fraction of the text is taken to be a
/// table of contents entry or a cross-reference, not the section itself.
const MIN_REFERENCES_POSITION: f64 = 0.3;

/// Headings that start a references section, lowercased with spaces removed.
const REFERENCES_HEADINGS: &[&str] = &[
    "references",
    "bibliography",
    "referencelist",
    "referencesandnotes",
    "literaturecited",
    "workscited",
];

#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    /// The `pdftotext` binary isn't on `PATH`.
    #[error("pdftotext not found on PATH (install poppler-utils)")]
    PdftotextMissing,
    /// `pdftotext` couldn't run, exited non-zero, or printed unreadable output.
    #[error("pdftotext failed: {0}")]
    Failed(String),
    /// The PDF has no text, e.g. a scanned PDF without a text layer.
    #[error("the PDF has no extractable text")]
    Empty,
    /// A binary the OCR path needs (`pdfinfo`, `pdftoppm` or `tesseract`)
    /// isn't on `PATH`.
    #[error(
        "{0} not found on PATH, so the PDF can't be OCRed (install poppler-utils and tesseract)"
    )]
    OcrToolMissing(&'static str),
}

/// Blocking: callers run it on `spawn_blocking`. References section stripped.
pub fn extract_text(pdf: &Path) -> Result<String, ExtractError> {
    extract_text_with(OsStr::new(PDFTOTEXT), pdf)
}

fn extract_text_with(program: &OsStr, pdf: &Path) -> Result<String, ExtractError> {
    // `-` writes to stdout; `-q` keeps its warnings about malformed PDFs,
    // which it recovers from, out of a failure's message.
    let output = Command::new(program)
        .args(["-q", "-enc", "UTF-8"])
        .arg(pdf)
        .arg("-")
        .output()
        .map_err(|e| match e.kind() {
            ErrorKind::NotFound => ExtractError::PdftotextMissing,
            _ => ExtractError::Failed(e.to_string()),
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        return Err(ExtractError::Failed(if stderr.is_empty() {
            format!("exited with {}", output.status)
        } else {
            stderr.to_string()
        }));
    }
    let raw = String::from_utf8(output.stdout)
        .map_err(|_| ExtractError::Failed("output isn't valid UTF-8".to_string()))?;
    let text = strip_references(&clean(&raw)).trim().to_string();
    if text.is_empty() {
        return Err(ExtractError::Empty);
    }
    Ok(text)
}

/// The PDF's page count, from poppler's `pdfinfo`. Blocking.
pub fn page_count(pdf: &Path) -> Result<usize, ExtractError> {
    page_count_with(OsStr::new(PDFINFO), pdf)
}

fn page_count_with(program: &OsStr, pdf: &Path) -> Result<usize, ExtractError> {
    let mut command = Command::new(program);
    command.arg(pdf);
    let stdout = run(&mut command, PDFINFO)?;
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("Pages:"))
        .and_then(|pages| pages.trim().parse().ok())
        .ok_or_else(|| ExtractError::Failed("pdfinfo printed no page count".to_string()))
}

/// Whether a text layer of `pages` pages is too sparse to stand for the
/// paper: under `MIN_CHARS_PER_PAGE` non-whitespace characters per page on
/// average, as a scan with only a typed cover page has.
pub fn is_sparse(text: &str, pages: usize) -> bool {
    let chars = text.chars().filter(|c| !c.is_whitespace()).count();
    chars < MIN_CHARS_PER_PAGE * pages
}

/// Blocking: callers run it on `spawn_blocking`. Reads every page of `pdf`
/// by OCR: each is rendered to a grayscale image at 300 dpi in a temporary
/// directory (removed afterwards) and read by `tesseract` as English, pages
/// in parallel up to `available_parallelism`. The text is cleaned and its
/// references section stripped as [`extract_text`]'s is.
pub fn ocr_text(pdf: &Path) -> Result<String, ExtractError> {
    ocr_text_with(&OcrTools::default(), pdf)
}

/// The binaries the OCR path runs, replaceable in tests.
struct OcrTools<'a> {
    pdfinfo: &'a OsStr,
    pdftoppm: &'a OsStr,
    tesseract: &'a OsStr,
}

impl Default for OcrTools<'_> {
    fn default() -> Self {
        OcrTools {
            pdfinfo: OsStr::new(PDFINFO),
            pdftoppm: OsStr::new(PDFTOPPM),
            tesseract: OsStr::new(TESSERACT),
        }
    }
}

fn ocr_text_with(tools: &OcrTools, pdf: &Path) -> Result<String, ExtractError> {
    let pages = page_count_with(tools.pdfinfo, pdf)?;
    let scratch = ScratchDir::new()?;
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .min(pages);
    // With several pages read at once, each tesseract keeps to one thread
    // rather than each starting one per core.
    let single_threaded = workers > 1;
    let texts: Vec<Mutex<Option<String>>> = (0..pages).map(|_| Mutex::new(None)).collect();
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let error = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= pages || failed.load(Ordering::Relaxed) {
                        break;
                    }
                    match ocr_page(tools, pdf, &scratch.0, index + 1, single_threaded) {
                        Ok(text) => *texts[index].lock().unwrap() = Some(text),
                        Err(e) => {
                            failed.store(true, Ordering::Relaxed);
                            error.lock().unwrap().get_or_insert(e);
                            break;
                        }
                    }
                }
            });
        }
    });
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    let raw = texts
        .into_iter()
        .map(|text| text.into_inner().unwrap().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\x0c");
    let text = strip_references(&clean(&raw)).trim().to_string();
    if text.is_empty() {
        return Err(ExtractError::Empty);
    }
    Ok(text)
}

/// Renders page `page` (1-based) of `pdf` into `dir` and reads it.
fn ocr_page(
    tools: &OcrTools,
    pdf: &Path,
    dir: &Path,
    page: usize,
    single_threaded: bool,
) -> Result<String, ExtractError> {
    let prefix = dir.join(format!("page-{page}"));
    let number = page.to_string();
    // `-singlefile` writes `<prefix>.png`, with no page number appended.
    let mut render = Command::new(tools.pdftoppm);
    render
        .args(["-q", "-f", &number, "-l", &number, "-r", OCR_DPI])
        .args(["-gray", "-png", "-singlefile"])
        .arg(pdf)
        .arg(&prefix);
    run(&mut render, PDFTOPPM)?;
    let image = prefix.with_extension("png");

    let mut read = Command::new(tools.tesseract);
    read.arg(&image).args(["-", "-l", "eng"]);
    if single_threaded {
        read.env("OMP_THREAD_LIMIT", "1");
    }
    let text = run(&mut read, TESSERACT);
    let _ = std::fs::remove_file(&image);
    text
}

/// Runs `command`, its stdout as UTF-8 (lossily: OCR output can hold stray
/// bytes); `tool` names the binary in a missing-binary error.
fn run(command: &mut Command, tool: &'static str) -> Result<String, ExtractError> {
    let output = command.output().map_err(|e| match e.kind() {
        ErrorKind::NotFound => ExtractError::OcrToolMissing(tool),
        _ => ExtractError::Failed(format!("{tool}: {e}")),
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stderr = stderr.trim();
        return Err(ExtractError::Failed(if stderr.is_empty() {
            format!("{tool} exited with {}", output.status)
        } else {
            format!("{tool}: {stderr}")
        }));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// A fresh temporary directory, removed with everything in it on drop.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Result<Self, ExtractError> {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.subsec_nanos());
        let dir = std::env::temp_dir().join(format!(
            "pax-ai-ocr-{}-{nanos}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir)
            .map_err(|e| ExtractError::Failed(format!("creating a temporary directory: {e}")))?;
        Ok(ScratchDir(dir))
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Turns page breaks into line breaks and rejoins words hyphenated across a
/// line break (`embed-\nding` → `embedding`). Only a hyphen between two
/// lowercase letters is rejoined, so `GPT-\n4` keeps its hyphen.
fn clean(raw: &str) -> String {
    let lines: Vec<&str> = raw.split(['\n', '\x0c']).map(str::trim_end).collect();
    let mut out = String::with_capacity(raw.len());
    for (i, line) in lines.iter().enumerate() {
        let next = lines.get(i + 1).map(|l| l.trim_start());
        let before_hyphen = line.strip_suffix('-').and_then(|l| l.chars().last());
        let rejoin = matches!(before_hyphen, Some(c) if c.is_lowercase())
            && matches!(next.and_then(|n| n.chars().next()), Some(c) if c.is_lowercase());
        if rejoin {
            out.push_str(&line[..line.len() - 1]);
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Cuts the text at its last references heading, if that heading is past
/// `MIN_REFERENCES_POSITION` of the text. Anything after it (appendices
/// included) goes too: telling an appendix from a long reference list isn't
/// reliable from plain text.
fn strip_references(text: &str) -> &str {
    let mut cut = None;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        if is_references_heading(line) {
            cut = Some(offset);
        }
        offset += line.len();
    }
    match cut {
        Some(at) if at as f64 >= text.len() as f64 * MIN_REFERENCES_POSITION => &text[..at],
        _ => text,
    }
}

/// A line holding only a references heading, optionally numbered (`7`,
/// `7.`, `VII.`) and letter-spaced the way small caps come out of
/// `pdftotext` (`R EFERENCES`).
fn is_references_heading(line: &str) -> bool {
    let mut tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() > 1 && is_section_number(tokens[0]) {
        tokens.remove(0);
    }
    let heading: String = tokens.concat().to_lowercase();
    let heading = heading.trim_end_matches([':', '.']);
    REFERENCES_HEADINGS.contains(&heading)
}

/// `7`, `7.`, `7.1`, `A.`, `VII`, `VII.`.
fn is_section_number(token: &str) -> bool {
    let bare = token.trim_end_matches(['.', ')']);
    if bare.is_empty() {
        return false;
    }
    let arabic = bare.chars().all(|c| c.is_ascii_digit() || c == '.');
    let roman = bare.chars().all(|c| "IVXLC".contains(c));
    let letter = bare.len() == 1 && bare.chars().all(|c| c.is_ascii_uppercase()) && bare != token;
    arabic || roman || letter
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::path::PathBuf;

    pub(crate) fn pdftotext_available() -> bool {
        Command::new(PDFTOTEXT)
            .arg("-v")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    /// A one-page PDF showing `lines`, with a correct xref table.
    pub(crate) fn pdf_with_lines(lines: &[&str]) -> Vec<u8> {
        let mut content = String::from("BT /F1 10 Tf 14 TL 72 760 Td\n");
        for line in lines {
            let escaped = line
                .replace('\\', "\\\\")
                .replace('(', "\\(")
                .replace(')', "\\)");
            content.push_str(&format!("({escaped}) Tj T*\n"));
        }
        content.push_str("ET");
        pdf_of_objects(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
              /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
                .to_vec(),
            stream_object("", content.as_bytes()),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        ])
    }

    /// A one-page PDF holding only a picture of `pdf_with_lines(lines)`,
    /// rendered at 150 dpi in a scratch dir named after `name`: no text
    /// layer, as a scanned paper has. `None` without `pdftoppm` to render it.
    pub(crate) fn image_only_pdf(name: &str, lines: &[&str]) -> Option<Vec<u8>> {
        let scratch = Scratch::new(&format!("image-{name}"));
        let text_pdf = scratch.file("text.pdf", &pdf_with_lines(lines));
        let prefix = scratch.0.join("page");
        let rendered = Command::new(PDFTOPPM)
            .args(["-q", "-r", "150", "-gray", "-singlefile"])
            .arg(&text_pdf)
            .arg(&prefix)
            .status()
            .is_ok_and(|s| s.success());
        if !rendered {
            return None;
        }
        // pdftoppm's PGM: `P5`, width, height and maxval, each followed by
        // one whitespace byte, then 8-bit gray pixels.
        let pgm = std::fs::read(prefix.with_extension("pgm")).ok()?;
        let mut fields = Vec::new();
        let mut start = 0;
        for (i, byte) in pgm.iter().enumerate() {
            if byte.is_ascii_whitespace() {
                fields.push(String::from_utf8_lossy(&pgm[start..i]).into_owned());
                start = i + 1;
                if fields.len() == 4 {
                    break;
                }
            }
        }
        let [magic, width, height, _maxval] = <[String; 4]>::try_from(fields).ok()?;
        assert_eq!(magic, "P5");
        let content = "q 612 0 0 792 0 0 cm /Im1 Do Q";
        Some(pdf_of_objects(&[
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
              /Resources << /XObject << /Im1 5 0 R >> >> /Contents 4 0 R >>"
                .to_vec(),
            stream_object("", content.as_bytes()),
            stream_object(
                &format!(
                    "/Type /XObject /Subtype /Image /Width {width} /Height {height} \
                     /ColorSpace /DeviceGray /BitsPerComponent 8 "
                ),
                &pgm[start..],
            ),
        ]))
    }

    /// A stream object: `dict` (its entries other than `/Length`) over `data`.
    fn stream_object(dict: &str, data: &[u8]) -> Vec<u8> {
        let mut object = format!("<< {dict}/Length {} >>\nstream\n", data.len()).into_bytes();
        object.extend_from_slice(data);
        object.extend_from_slice(b"\nendstream");
        object
    }

    /// A PDF of `objects`, numbered from 1 (the first is the catalog), with
    /// a correct xref table.
    fn pdf_of_objects(objects: &[Vec<u8>]) -> Vec<u8> {
        let mut pdf = b"%PDF-1.4\n".to_vec();
        let mut offsets = Vec::new();
        for (i, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
            pdf.extend_from_slice(object);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let xref = pdf.len();
        let mut tail = format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1);
        for offset in offsets {
            tail.push_str(&format!("{offset:010} 00000 n \n"));
        }
        tail.push_str(&format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        ));
        pdf.extend_from_slice(tail.as_bytes());
        pdf
    }

    /// Whether the OCR path's binaries are on `PATH`, tesseract with its
    /// English data.
    pub(crate) fn ocr_available() -> bool {
        let runs = |program: &str| {
            Command::new(program)
                .arg("-v")
                .output()
                .is_ok_and(|o| o.status.success())
        };
        let english = Command::new(TESSERACT)
            .arg("--list-langs")
            .output()
            .is_ok_and(|o| {
                o.status.success()
                    && String::from_utf8_lossy(&o.stdout)
                        .lines()
                        .any(|line| line.trim() == "eng")
            });
        runs(PDFINFO) && runs(PDFTOPPM) && english
    }

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("pax-ai-extract-{name}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }

        fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn extracts_the_body_and_strips_the_references() {
        if !pdftotext_available() {
            return;
        }
        let scratch = Scratch::new("body");
        let pdf = scratch.file(
            "paper.pdf",
            &pdf_with_lines(&[
                "Introduction",
                "Retrieval augmented generation grounds a model in documents.",
                "Methods",
                "We embed every chunk of every paper.",
                "Results",
                "Chunk set similarity beats the mean vector.",
                "References",
                "Smith, J. Some cited paper. 2020.",
            ]),
        );
        let text = extract_text(&pdf).unwrap();
        assert!(text.starts_with("Introduction"), "{text}");
        assert!(text.contains("We embed every chunk of every paper."));
        assert!(text.ends_with("Chunk set similarity beats the mean vector."));
        assert!(!text.contains("References"));
        assert!(!text.contains("Smith"));
    }

    #[test]
    fn a_pdf_without_text_is_empty() {
        if !pdftotext_available() {
            return;
        }
        let scratch = Scratch::new("empty");
        let pdf = scratch.file("blank.pdf", &pdf_with_lines(&[]));
        assert!(matches!(extract_text(&pdf), Err(ExtractError::Empty)));
    }

    #[test]
    fn a_file_that_isnt_a_pdf_fails() {
        if !pdftotext_available() {
            return;
        }
        let scratch = Scratch::new("garbage");
        let pdf = scratch.file("notes.pdf", b"not a pdf at all");
        assert!(matches!(extract_text(&pdf), Err(ExtractError::Failed(_))));
    }

    #[test]
    fn a_missing_file_fails() {
        if !pdftotext_available() {
            return;
        }
        let scratch = Scratch::new("missing");
        let pdf = scratch.0.join("nowhere.pdf");
        assert!(matches!(extract_text(&pdf), Err(ExtractError::Failed(_))));
    }

    #[test]
    fn a_missing_binary_is_reported_as_such() {
        let missing = OsStr::new("pax-ai-no-such-pdftotext");
        assert!(matches!(
            extract_text_with(missing, Path::new("paper.pdf")),
            Err(ExtractError::PdftotextMissing)
        ));
    }

    /// A short scanned paper: its body, then a references section.
    const SCANNED: &[&str] = &[
        "Introduction",
        "Scanned papers carry their words only as pictures of text.",
        "Methods",
        "Optical character recognition reads every rendered page.",
        "References",
        "Smith, J. Some cited paper. 2020.",
    ];

    #[test]
    fn ocr_reads_an_image_only_pdf_and_strips_the_references() {
        if !ocr_available() || !pdftotext_available() {
            eprintln!("skipped: pdftotext, pdfinfo, pdftoppm or tesseract isn't available");
            return;
        }
        let scratch = Scratch::new("ocr");
        let pdf = scratch.file("scan.pdf", &image_only_pdf("ocr", SCANNED).unwrap());
        assert!(matches!(extract_text(&pdf), Err(ExtractError::Empty)));
        assert_eq!(page_count(&pdf).unwrap(), 1);

        let text = ocr_text(&pdf).unwrap();
        assert!(text.starts_with("Introduction"), "{text}");
        for words in ["pictures of text", "Optical character recognition"] {
            assert!(text.contains(words), "{words:?} in {text:?}");
        }
        assert!(!text.contains("Smith"), "{text}");
    }

    #[test]
    fn ocr_of_a_blank_page_is_empty() {
        if !ocr_available() {
            eprintln!("skipped: pdfinfo, pdftoppm or tesseract isn't available");
            return;
        }
        let scratch = Scratch::new("ocr-blank");
        let pdf = scratch.file("blank.pdf", &image_only_pdf("ocr-blank", &[]).unwrap());
        assert!(matches!(ocr_text(&pdf), Err(ExtractError::Empty)));
    }

    #[test]
    fn a_missing_ocr_binary_is_reported_by_name() {
        let missing = OsStr::new("pax-ai-no-such-binary");
        let tools = OcrTools {
            pdfinfo: missing,
            ..OcrTools::default()
        };
        assert!(matches!(
            ocr_text_with(&tools, Path::new("paper.pdf")),
            Err(ExtractError::OcrToolMissing("pdfinfo"))
        ));

        if !ocr_available() {
            return;
        }
        let scratch = Scratch::new("ocr-missing");
        let pdf = scratch.file("scan.pdf", &pdf_with_lines(&["One page."]));
        for (tools, tool) in [
            (
                OcrTools {
                    pdftoppm: missing,
                    ..OcrTools::default()
                },
                "pdftoppm",
            ),
            (
                OcrTools {
                    tesseract: missing,
                    ..OcrTools::default()
                },
                "tesseract",
            ),
        ] {
            let result = ocr_text_with(&tools, &pdf);
            assert!(
                matches!(result, Err(ExtractError::OcrToolMissing(t)) if t == tool),
                "{tool}: {result:?}"
            );
        }
    }

    #[test]
    fn a_text_layer_under_100_characters_a_page_is_sparse() {
        let cover = "A Scanned Paper\nJ. Smith\n1987";
        assert!(is_sparse(cover, 12));
        assert!(!is_sparse(cover, 0));
        let page = "x".repeat(MIN_CHARS_PER_PAGE);
        assert!(!is_sparse(&page, 1));
        assert!(is_sparse(&page, 2));
        // Whitespace doesn't count.
        assert!(is_sparse(&format!("{}\n\n", " ".repeat(500)), 1));
    }

    #[test]
    fn rejoins_words_hyphenated_across_lines() {
        assert_eq!(
            clean("an embed-\nding of GPT-\n4\x0cnext page\n"),
            "an embedding of GPT-\n4\nnext page\n\n"
        );
    }

    #[test]
    fn recognizes_references_headings() {
        for heading in [
            "References",
            "REFERENCES",
            "References:",
            "7 References",
            "7. References",
            "VII. REFERENCES",
            "R EFERENCES",
            "Bibliography",
            "Literature Cited",
            "  References  \n",
        ] {
            assert!(is_references_heading(heading), "{heading:?}");
        }
        for line in [
            "See the references below.",
            "References are listed at the end",
            "Referenced work",
            "",
        ] {
            assert!(!is_references_heading(line), "{line:?}");
        }
    }

    #[test]
    fn cuts_at_the_last_references_heading() {
        let text = "Contents\nReferences\nIntroduction\nbody body body body\n\
                    References\nSmith 2020\nAppendix\n";
        assert_eq!(
            strip_references(text),
            "Contents\nReferences\nIntroduction\nbody body body body\n"
        );
    }

    #[test]
    fn keeps_a_references_heading_near_the_start() {
        let text = "References\nthe whole paper follows, long enough to put the \
                    heading well inside the first part of the text\n";
        assert_eq!(strip_references(text), text);
    }

    #[test]
    fn keeps_text_without_a_references_heading() {
        let text = "Introduction\nbody\n";
        assert_eq!(strip_references(text), text);
    }
}
