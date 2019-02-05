library("forecast")
library("TTR")

plotForecastErrors <- function(forecasterrors) {
     # make a histogram of the forecast errors:
     mybinsize <- IQR(forecasterrors, na.rm=TRUE)/4
     mysd   <- sd(forecasterrors, na.rm=TRUE)
     mymin  <- min(forecasterrors, na.rm=TRUE) - mysd*5
     mymax  <- max(forecasterrors, na.rm=TRUE) + mysd*3
     # generate normally distributed data with mean 0 and standard deviation mysd
     mynorm <- rnorm(10000, mean=0, sd=mysd)
     mymin2 <- min(mynorm)
     mymax2 <- max(mynorm)
     if (mymin2 < mymin) { mymin <- mymin2 }
     if (mymax2 > mymax) { mymax <- mymax2 }
     # make a red histogram of the forecast errors, with the normally distributed data overlaid:
     mybins <- seq(mymin, mymax, mybinsize)
     hist(forecasterrors, col="red", freq=FALSE, breaks=mybins)
     # freq=FALSE ensures the area under the histogram = 1
     # generate normally distributed data with mean 0 and standard deviation mysd
     myhist <- hist(mynorm, plot=FALSE, breaks=mybins)
     # plot the normal curve as a blue line on top of the histogram of forecast errors:
     points(myhist$mids, myhist$density, type="l", col="blue", lwd=2)
  }



ea <- scan("list8/EA.dat")
eats <- ts(ea)

plot(eats)

##EAcomponents <- decompose(eats)

eatsSMA <- SMA(eats, n=4)
plot.ts(eatsSMA)

## brak sezonowości
eaForecast <- HoltWinters(eats, gamma=FALSE)
eaForecast
plot(eaForecast)

eaForecast2 <- forecast(eaForecast, h=10)
plot(eaForecast2)

acf(eaForecast2$residuals, lag.max=20, na.action=na.pass)
Box.test(eaForecast2$residuals, lag=20, type="Ljung-Box")

plot.ts(eaForecast2$residuals)
plotForecastErrors(eaForecast2$residuals)

### bez bety

## brak sezonowości
eaForecast <- HoltWinters(eats, beta=FALSE, gamma=FALSE)
eaForecast
plot(eaForecast)

eaForecast2 <- forecast(eaForecast, h=10)
plot(eaForecast2)

acf(eaForecast2$residuals, lag.max=20, na.action=na.pass)
Box.test(eaForecast2$residuals, lag=20, type="Ljung-Box")

plot.ts(eaForecast2$residuals)
plotForecastErrors(eaForecast2$residuals)



#### ARIMA

eadiff <- diff(eats, differences=2)
plot.ts(eadiff) ## no nie wiem



acf(eadiff, lag.max=20)  
acf(eadiff, lag.max=20, plot=FALSE)  ## 1

pacf(eadiff, lag.max=20)  
pacf(eadiff, lag.max=20, plot=FALSE) ## 2


eaarima <- arima(eats, order=c(1,2,1)) 
eaarima

eaarimaforecast <- forecast(eaarima, h=10)
plot(eaarimaforecast)

acf(eaarimaforecast$residuals, lag.max=20)
Box.test(eaarimaforecast$residuals, lag=20, type="Ljung-Box")

plot.ts(eaarimaforecast$residuals)
plotForecastErrors(eaarimaforecast$residuals)
