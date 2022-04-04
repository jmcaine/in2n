
const BUFFER_SIZE: usize = 88200; // enough to hold 2 second of buffer @44k sample-rate. (Normally, we'll process about a tenth of a second at a time, so this is more than ample.)
//const BUFFER_SIZE: usize = 34053; // for testing
//const BUFFER_SIZE: usize = 44100; // enough to hold 1 second of buffer @44k sample-rate. (Normally, we'll process about a tenth of a second at a time, so this is more than ample.)
//const SAMPLES_PER_FFT: usize = 16384; // that is, 2^14
const SAMPLES_PER_FFT: usize = 8192; // that is, 2^13 - slightly less than 1/4 of a second (at 44k sample rate, specified above), and a power of two (for FFT) - G3 is 196 cycles/second; this (2^13) works out to .186 seconds, which is about 36 peaks per frame at that lowest frequency, which should still be plenty for an FFT; obviously, as pitch goes up, so does the suitability of this small sample frame, for FFT purposes, because there will be more peaks per frame at higher frequencies.
//const SAMPLES_PER_FFT: usize = 4096; // that is, 2^12
//const SAMPLES_PER_FFT: usize = 1024; // that is, 2^10; 1024 is considered a good balance between efficiency and accuracy for audio purposes (NOTE: may need to increase this for accuracy at lower notes!)
const TOP_SCORE: u8 = 100;

pub struct Setup {
	samples_per_second: f32,
}

impl Setup {
	fn new(samples_per_second: f32) -> Setup {
		Setup { samples_per_second: samples_per_second, }
	}
}
