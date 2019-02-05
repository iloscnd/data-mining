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



goog <- scan("list8/GOOG.dat")
googts <- ts(goog)

plot(googts)

##GOOGcomponents <- decompose(googts)

googtsSMA <- SMA(googts, n=5)
plot.ts(googtsSMA)

## brak sezonowości
googForecast <- HoltWinters(googts, gamma=FALSE)
googForecast
plot(googForecast)

googForecast2 <- forecast(googForecast, h=10)
plot(googForecast2)

acf(googForecast2$residuals, lag.max=20, na.action=na.pass)
Box.test(googForecast2$residuals, lag=20, type="Ljung-Box")

plot.ts(googForecast2$residuals)
plotForecastErrors(googForecast2$residuals)


#### ARIMA

googdiff <- diff(googts, differences=1)
plot.ts(googdiff) ## with one looks ok



acf(googdiff, lag.max=20)  
acf(googdiff, lag.max=20, plot=FALSE)  

pacf(googdiff, lag.max=20)  
pacf(googdiff, lag.max=20, plot=FALSE) 

## xDDDD arima(0,0)

googarima <- arima(googts, order=c(0,1,0)) ## random walk
googarima

googarimaforecast <- forecast(googarima, h=10)
plot(googarimaforecast)

acf(googarimaforecast$residuals, lag.max=20)
Box.test(googarimaforecast$residuals, lag=20, type="Ljung-Box")

plot.ts(googarimaforecast$residuals)
plotForecastErrors(googarimaforecast$residuals)