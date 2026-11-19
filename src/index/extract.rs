//! A paper's full text, extracted from its PDF with poppler's `pdftotext`.
//!
//! The references section is stripped: it's a list of other papers' titles,
//! which would make every paper look similar to the ones it cites.

use std::ffi::OsStr;
use std::io::ErrorKind;
use std::path::Path;
use std::process::Command;

/// Part of every full-text store key: bump it when a change here alters the
/// text extracted from the same PDF, so old entries read as misses.
pub const EXTRACTOR_VERSION: u32 = 1;

/// The extractor's binary, looked up on `PATH` (the flake's `pax-mcp` wraps
/// poppler onto it).
const PDFTOTEXT: &str = "pdftotext";

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
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn pdftotext_available() -> bool {
        Command::new(PDFTOTEXT)
            .arg("-v")
            .output()
            .is_ok_and(|o| o.status.success())
    }

    /// A one-page PDF showing `lines`, with a correct xref table.
    fn pdf_with_lines(lines: &[&str]) -> Vec<u8> {
        let mut content = String::from("BT /F1 10 Tf 14 TL 72 760 Td\n");
        for line in lines {
            let escaped = line
                .replace('\\', "\\\\")
                .replace('(', "\\(")
                .replace(')', "\\)");
            content.push_str(&format!("({escaped}) Tj T*\n"));
        }
        content.push_str("ET");
        let objects = [
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] \
             /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
                .to_string(),
            format!(
                "<< /Length {} >>\nstream\n{content}\nendstream",
                content.len()
            ),
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        ];
        let mut pdf = String::from("%PDF-1.4\n");
        let mut offsets = Vec::new();
        for (i, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            pdf.push_str(&format!("{} 0 obj\n{object}\nendobj\n", i + 1));
        }
        let xref = pdf.len();
        pdf.push_str(&format!(
            "xref\n0 {}\n0000000000 65535 f \n",
            objects.len() + 1
        ));
        for offset in offsets {
            pdf.push_str(&format!("{offset:010} 00000 n \n"));
        }
        pdf.push_str(&format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        ));
        pdf.into_bytes()
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
