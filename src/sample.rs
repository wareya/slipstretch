#[derive(Clone, Copy, Debug, Default)]
pub (crate) struct Sample
{
    pub (crate) l: f32,
    pub (crate) r: f32,
}
impl Sample
{
    pub (crate) fn energy_sq(&self) -> f32
    {
        self.l*self.l + self.r*self.r
    }
}
impl core::ops::Add<Sample> for Sample
{
    type Output = Sample;
    fn add(self, other: Sample) -> Sample
    {
        Sample { l: self.l + other.l, r: self.r + other.r }
    }
}
impl core::ops::Sub<Sample> for Sample
{
    type Output = Sample;
    fn sub(self, other: Sample) -> Sample
    {
        Sample { l: self.l - other.l, r: self.r - other.r }
    }
}
impl core::ops::Mul<f32> for Sample
{
    type Output = Sample;
    fn mul(self, scalar: f32) -> Sample
    {
        Sample { l: self.l * scalar, r: self.r * scalar }
    }
}
impl core::ops::Div<f32> for Sample
{
    type Output = Sample;
    fn div(self, divisor: f32) -> Sample
    {
        Sample { l: self.l / divisor, r: self.r / divisor }
    }
}
impl core::ops::AddAssign<Sample> for Sample
{
    fn add_assign(&mut self, other: Sample)
    {
        self.l += other.l;
        self.r += other.r;
    }
}
impl core::ops::SubAssign<Sample> for Sample
{
    fn sub_assign(&mut self, other: Sample)
    {
        self.l -= other.l;
        self.r -= other.r;
    }
}

impl core::ops::MulAssign<f32> for Sample
{
    fn mul_assign(&mut self, scalar: f32)
    {
        self.l *= scalar;
        self.r *= scalar;
    }
}

impl core::ops::DivAssign<f32> for Sample
{
    fn div_assign(&mut self, divisor: f32)
    {
        self.l /= divisor;
        self.r /= divisor;
    }
}
