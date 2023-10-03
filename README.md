# slipstretch
slipstretch is a simple experimental audio stretching algorithm that splits the audio stream up into frequency bands before stretching them. Then, it stretches each frequency band separately. This allows higher frequencies to use a smaller window/chunk size, reducing transient[^3] flutter[^1]/flam[^2] artifacts. The one cross-band constraint[^4] is that lower frequency bands are not allowed to timeshift further "left" than higher frequency bands, which reduces bass smearing[^5].

[^1]: flutter -> warble but in high frequencies instead of low
[^2]: flam -> what drummers call it when you get two transients right next to eachother instead of lined up on top of each other
[^3]: transient -> when you have click or tap or smash sounds instead of smooth rise/fall
[^4]: cross-band constraint -> not jargon, means what the individual parts mean, it's a constraint applied across frequency bands when doing the time shifting
[^5]: bass smearing -> when bass frequencies of drums come before their high-frequency attack instead of after; sounds really bad

The underlying audio stretching algorithm after splitting the audio stream up into frequency bands is very simple, so it doesn't sound very good. If a better underlying algorithm were used, then this would be much better.

## Examples

0.75x and 1.5x pitch shift music example (make sure to unmute it):

https://github.com/wareya/slipstretch/assets/585488/4ade9eef-dcc3-4bcb-a515-42c2834f7b60

110% time stretch music example (multiband -> fullband -> original) (make sure to unmute it):

https://github.com/wareya/slipstretch/assets/585488/6955189c-ed54-44dd-b637-162d1dc841aa

For more, see the [examples directory](https://github.com/wareya/slipstretch/tree/main/example).
