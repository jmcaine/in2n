
use crate::setup;
use crate::setup::Setup;
use crate::freqs;

pub struct Note {
	index: usize, // index into freqs::{temperament-array} - see freqs.rs
	duration: u16, // ms
}

impl Note {
	fn new(index: usize, duration: u16) -> Note {
		Note {
			index: index,
			duration: duration,
		}
	}
}

// ---------------------------------------------

pub struct Score<'a> {
	note: &'a Note,
	score: u8,
}

impl Score<'_> {
	fn new<'b>(note: &'b Note) -> Score<'b> {
		Score {
			note: note,
			score: setup::TOP_SCORE,
		}
	}

	fn decrement(&mut self, divisions: usize) {
		self.score -= setup::TOP_SCORE / divisions;
	}

}

// ---------------------------------------------

pub struct Song {
	notes: Vec<Note>,
	cp: usize, // current position in the song; index into  notes, indicating the "current" (target) note (note being presently played, or, perhaps, note about to be played
	cp_duration: u16, // how long, in ms, the 'cp' has been underway
	setup: &Setup,
	temperament: usize,

	hit_frames: u16, // number of "hits" (frames which we've evaluated to a "match" between incoming frequency and expected frequency at song's cp), for the current note (i.e., cp; since the hit-detection started for this note); silent fames are not counted
		//TODO: should be cp_hit_frames; consider abstracting out cp, to contain: duration, frames, hit_frames, silent_frames....
	cp_frames: u16, // the number of frames attempted (during the processing of the current note) (not including silent frames
	silent_frames: u16, // the number of frames (during the processing of the current note) that appear to be "silent"
	pitch_precision: f32, // deviation of input frequency from expected frequency
}

impl Song {
	fn new(notes: Vec::<Note>, setup: &Setup, temperament: usize) -> Song {
		Song {
			notes: notes,
			setup: setup,
			temperament: temperament,
			cp: 0,
			cp_duration: 0,
		}
	}
	fn match_note<'c>(&mut self, note: &'c Note, frequencies: &[f32]) -> Score<'c> { // trial bins of frequencies, so we get the most powerful and, possibly, some octaves; it may be that the real root is less powerful than a octave harmonic, so this may do nothing....
		let temperament = freqs::TEMPERAMENTS[self.temperament];
		let target = temperament[note.index];
		let below = if note.index > 0 { temperament[note.index - 1]  } else { target };
		let above = if note.index + 1 < temperament.len() { temperament[note.index + 1] } else { target };
		let low_bound = below + (target - below) * 2./3.; // TODO: instead, consider note before or notes around (within our song note sequence)-- you might be able to set this low lower and catch "way too flat" input without risking being on the wrong note in the sequence
		let high_bound = above - (above - target) * 2./3.;
		println!("lo: {}, hi: {}, actual: {} {} {}", low_bound, high_bound, frequencies[0], frequencies[1], frequencies[2]);
		let mut score = Score::new(&note);
		for frequency in frequencies.iter() {
			if low_bound > *frequency || *frequency > high_bound { // TODO: what if one of the frequency values is of a harmonic? Then we'd have to check against low/high-bounds of the harmonics!?
				score.decrement(frequencies.len());
			} else {
				break;
			}
		}
		// else...
		return score;
	}

	fn match_current_note(&mut self, frequencies: &[f32]) -> f32 {
		let primary = self.match_note(self.notes[self.cp], &frequencies);
		if primary.score > setup::TOP_SCORE / 2 {
			self.cp_duration += 1000 * (setup::SAMPLES_PER_FFT as f32 / self.samples_per_second) as u16;
			return freqs::TEMPERAMENTS[self.temperament][self.notes[self.cp].index];
		} else {
			// Check for other candidates, like previous note, next note, back to note at the start of the measure, etc., then, if not, decide whether to continue to fill with self.cp note or silence....
			self.cp_duration = 0; // TODO ???
			return 0f32; // "silence" indicator
		}
	}
}
