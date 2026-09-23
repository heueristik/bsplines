//! Provides short working buffers for the recurrences.

/// Calls the calculation with a zeroed buffer of the given length. A buffer of up to 16 values lives
/// on the stack, so the common low degrees need no allocation.
pub(crate) fn with_buffer<T>(length: usize, calculate: impl FnOnce(&mut [f64]) -> T) -> T {
    let mut stack_buffer = [0.0; 16];
    if length <= stack_buffer.len() {
        calculate(&mut stack_buffer[..length])
    } else {
        calculate(&mut vec![0.0; length])
    }
}
