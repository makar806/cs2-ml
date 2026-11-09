"""Independent scalar 1-Euro filter (Casiez et al., CHI 2012 equations)."""
import math

class OneEuro:
    def __init__(self,min_cutoff=1.,beta=.01,derivative_cutoff=1.):
        if not all(math.isfinite(v) for v in (min_cutoff,beta,derivative_cutoff)) or min_cutoff<=0 or beta<0 or derivative_cutoff<=0:
            raise ValueError('Invalid filter parameters')
        self.min_cutoff=min_cutoff;self.beta=beta;self.derivative_cutoff=derivative_cutoff
        self.reset()
    def reset(self):
        self.at=None;self.raw=None;self.value=None;self.derivative=0.
    @staticmethod
    def alpha(cutoff,dt):
        return 1/(1+1/(2*math.pi*cutoff*dt))
    def __call__(self,value,at):
        if not math.isfinite(value) or not math.isfinite(at):raise ValueError('Nonfinite observation')
        if self.at is None:
            self.at=at;self.raw=value;self.value=value
            return value
        if at<=self.at:raise ValueError('Source timestamps must increase')
        dt=at-self.at
        derivative=(value-self.raw)/dt
        a=self.alpha(self.derivative_cutoff,dt)
        self.derivative+=a*(derivative-self.derivative)
        a=self.alpha(self.min_cutoff+self.beta*abs(self.derivative),dt)
        self.value+=a*(value-self.value)
        self.raw=value;self.at=at
        return self.value
