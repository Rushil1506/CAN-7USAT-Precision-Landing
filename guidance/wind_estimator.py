class WindEstimator:
    """
    Exponential Moving Average (EMA) wind estimator.
    
    Provides a shared filtered wind estimate for all guidance components
    to prevent high-frequency control chatter.
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.wind_x = 0.0
        self.wind_y = 0.0
        self.initialized = False
        
    def update(self, raw_x: float, raw_y: float) -> None:
        if not self.initialized:
            self.wind_x = raw_x
            self.wind_y = raw_y
            self.initialized = True
        else:
            self.wind_x = self.alpha * raw_x + (1.0 - self.alpha) * self.wind_x
            self.wind_y = self.alpha * raw_y + (1.0 - self.alpha) * self.wind_y
            
    def reset(self) -> None:
        self.wind_x = 0.0
        self.wind_y = 0.0
        self.initialized = False
