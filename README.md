# slipstretch
slipstretch is a simple experimental audio stretching algorithm that splits the audio stream up into frequency bands before stretching them. Then, it stretches each frequency band separately. This allows higher frequencies to use a smaller window/chunk size, reducing transient flutter artifacts. The one cross-band constraint is that lower frequency bands are not allowed to timeshift further "left" than higher frequency bands, which reduces bass smearing.

The underlying audio stretching algorithm after splitting the audio stream up into frequency bands is very simple, so it doesn't sound very good. If a better underlying algorithm were used, then this would be much better.

## Examples

See the [examples directory](https://github.com/wareya/slipstretch/tree/main/example).