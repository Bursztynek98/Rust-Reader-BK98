//! Text change detection utilities (Port of diff_utils.py).
//! Uses normalized character mapping and Levenshtein ratio to filter out OCR misreads.

use strsim::levenshtein;

const CHANGE_THRESHOLD: f64 = 0.70;
const MIN_TEXT_LENGTH: usize = 3;

/// Normalize text: lowercase, apply OCR misread character mapping, collapse whitespace.
pub fn normalize_text(text: &str) -> String {
    let mut text = text.to_lowercase();
    
    // Replace common misread ASCII chars with Polish/Latin equivalents
    let norm_map = [
        ('$', 'ś'),
        ('§', 'ś'),
        ('^', 'ą'),
        ('~', 'ż'),
        ('#', 'ń'),
        ('@', 'ę'),
        ('&', 'ź'),
        ('|', 'l'),
        ('!', 'i'),
        ('0', 'o'),
        ('1', 'l'),
        ('3', 'e'),
        ('4', 'a'),
        ('5', 's'),
        ('8', 'b'),
    ];

    for (src, dst) in norm_map {
        text = text.replace(src, &dst.to_string());
    }

    // Collapse whitespace
    let words: Vec<&str> = text.split_whitespace().collect();
    words.join(" ")
}

/// Calculate Levenshtein similarity ratio between two strings:
/// ratio = (len1 + len2 - edit_distance) / (len1 + len2)
pub fn levenshtein_ratio(s1: &str, s2: &str) -> f64 {
    let c1: Vec<char> = s1.chars().collect();
    let c2: Vec<char> = s2.chars().collect();
    let len1 = c1.len();
    let len2 = c2.len();

    if len1 == 0 && len2 == 0 {
        return 1.0;
    }
    let total_len = len1 + len2;
    if total_len == 0 {
        return 1.0;
    }

    let dist = levenshtein(s1, s2);
    if dist >= total_len {
        return 0.0;
    }

    (total_len - dist) as f64 / total_len as f64
}

/// Returns true if `new_text` differs significantly from `old_text`
/// to warrant a new TTS audio generation.
pub fn has_significant_change(old_text: &str, new_text: &str) -> bool {
    let old_n = normalize_text(old_text);
    let new_n = normalize_text(new_text);

    if old_n.is_empty() && new_n.is_empty() {
        return false;
    }

    if new_n.is_empty() {
        return false; // Don't trigger on transient blank frames
    }

    if new_n.chars().count() < MIN_TEXT_LENGTH {
        return false; // Ignore single chars / noise
    }

    if old_n.is_empty() {
        return true; // Fresh subtitle after silence
    }

    let ratio = levenshtein_ratio(&old_n, &new_n);
    ratio < CHANGE_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization() {
        assert_eq!(normalize_text("  W!taj  $wiecie  "), "witaj świecie");
    }

    #[test]
    fn test_identical_text() {
        assert!(!has_significant_change("Cześć jak się masz?", "Cześć jak się masz?"));
    }

    #[test]
    fn test_diacritic_misread_ignored() {
        // Minor difference due to misread should ratio >= 0.70 -> false
        assert!(!has_significant_change("Cześć jak się masz?", "Cze$c jak s!e masz?"));
    }

    #[test]
    fn test_new_subtitle() {
        assert!(has_significant_change("Cześć jak się masz?", "Gdzie idziemy jutro rano?"));
    }
}
