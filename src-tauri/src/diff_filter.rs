//! Text change detection utilities (Port of diff_utils.py).
//! Uses normalized character mapping and Levenshtein ratio to filter out OCR misreads.

use strsim::levenshtein;

const CHANGE_THRESHOLD: f64 = 0.70;
const MIN_TEXT_LENGTH: usize = 3;

/// Normalize text: lowercase, apply OCR misread character mapping, collapse whitespace.
pub fn normalize_text(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut in_whitespace = false;

    for ch in text.chars() {
        let lower = ch.to_lowercase().next().unwrap_or(ch);
        let mapped = match lower {
            '$' | '§' => 'ś',
            '^' => 'ą',
            '~' => 'ż',
            '#' => 'ń',
            '@' => 'ę',
            '&' => 'ź',
            '|' | '1' => 'l',
            '!' => 'i',
            '0' => 'o',
            '3' => 'e',
            '4' => 'a',
            '5' => 's',
            '8' => 'b',
            c => c,
        };

        if mapped.is_whitespace() {
            if !in_whitespace && !normalized.is_empty() {
                normalized.push(' ');
                in_whitespace = true;
            }
        } else {
            normalized.push(mapped);
            in_whitespace = false;
        }
    }

    if normalized.ends_with(' ') {
        normalized.pop();
    }

    normalized
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

/// 256-bit Average Hash (aHash) for fast frame difference detection (<0.1ms)
pub type FrameHash = [u64; 4];

pub fn compute_frame_hash(img: &image::DynamicImage) -> FrameHash {
    use image::imageops;
    let resized = img.resize_exact(16, 16, imageops::FilterType::Nearest);
    let gray = resized.to_luma8();
    let pixels = gray.as_raw();

    let sum: u32 = pixels.iter().map(|&p| p as u32).sum();
    let avg = (sum / 256) as u8;

    let mut hash = [0u64; 4];
    for (i, &p) in pixels.iter().enumerate() {
        if p >= avg {
            let chunk = i / 64;
            let bit = i % 64;
            hash[chunk] |= 1u64 << bit;
        }
    }
    hash
}

/// Returns true if the captured image changed significantly compared to `prev_hash`.
pub fn has_image_changed(prev_hash: Option<FrameHash>, current_img: &image::DynamicImage) -> (bool, FrameHash) {
    let current_hash = compute_frame_hash(current_img);
    let prev = match prev_hash {
        Some(h) => h,
        None => return (true, current_hash),
    };

    let diff_bits: u32 = prev.iter()
        .zip(current_hash.iter())
        .map(|(&a, &b)| (a ^ b).count_ones())
        .sum();

    // If Hamming distance > 6 bits out of 256 (~2.3% difference), image has changed
    (diff_bits > 6, current_hash)
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
