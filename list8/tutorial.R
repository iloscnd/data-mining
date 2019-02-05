library("TTR")
library("forecast")

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

## Kings - smooth moving avarage

kings <- scan("list8/kings.dat", skip=3)
kingsts <- ts(kings)

plot.ts(kingsts)

kingstsSMA3 <- SMA(kingsts)
plot.ts(kingstsSMA3)

kingstsSMA8 <- SMA(kingsts,n=8)
plot.ts(kingstsSMA8)


## births - decomposition

births <- scan("list8/nybirths.dat")
birthsts <- ts(births, frequency=12, start=c(1946,1))

plot.ts(birthsts)

birthstscomponents <- decompose(birthsts)

plot(birthstscomponents)

birthstscomponentsseason <- birthsts - birthstscomponents$seasonal

plot(birthstscomponentsseason)



##Rain - only level

rain <- scan("list8/precip1.dat",skip=1)

rainseries <- ts(rain,start=c(1813))
plot.ts(rainseries)

rainseriesforecasts <- HoltWinters(rainseries, beta=FALSE, gamma=FALSE)

rainseriesforecasts$fitted

plot(rainseriesforecasts)

rainseriesforecasts$SSE # sum of squared errors

rainseriesforecasts2 <- forecast(rainseriesforecasts, h=8)

plot(rainseriesforecasts2)

#correlogram

acf(rainseriesforecasts2$residuals, lag.max=20, na.action=na.pass)

Box.test(rainseriesforecasts2$residuals, lag=20, type="Ljung-Box")


plotForecastErrors(rainseriesforecasts2$residuals)

## skirts = levels and slope

skirts <- scan("list8/skirts.dat", skip=5)

skirtsseries <- ts(skirts, start=c(1866))
plot.ts(skirtsseries)

skirtsseriesforecasts <- HoltWinters(skirtsseries, gamma=FALSE)

skirtsseriesforecasts
skirtsseriesforecasts$SSE

plot(skirtsseriesforecasts)

skirtsseriesforecasts2 <- forecast(skirtsseriesforecasts, h=19)
plot(skirtsseriesforecasts2)

acf(skirtsseriesforecasts2$residuals, lag.max=20, na.action=na.pass)
Box.test(skirtsseriesforecasts2$residuals, lag=20, type="Ljung-Box")

plot.ts(skirtsseriesforecasts2$residuals)
plotForecastErrors(skirtsseriesforecasts2$residuals)

## souvenir - level, slope and seasionality

souvenir <- scan("list8/fancy.dat")
souvenirts <- ts(souvenir, frequency=12, start=c(1987,1))

plot.ts(souvenirts)

logsouvenirts <- log(souvenirts)
plot.ts(logsouvenirts)

souvenirtimeseriesforecasts <- HoltWinters(logsouvenirts)
souvenirtimeseriesforecasts
souvenirtimeseriesforecasts$SSE
plot(souvenirtimeseriesforecasts)

souvenirtimeseriesforecasts2 <- forecast(souvenirtimeseriesforecasts, h=48)
plot(souvenirtimeseriesforecasts2)

acf(souvenirtimeseriesforecasts2$residuals, lag.max=20, na.action=na.pass)
Box.test(souvenirtimeseriesforecasts2$residuals, lag=20, type="Ljung-Box")

plot.ts(souvenirtimeseriesforecasts2$residuals)            
plotForecastErrors(souvenirtimeseriesforecasts2$residuals)


### ARIMA

## kings 
## d - number of differentation
kingtimeseriesdiff1 <- diff(kingsts, differences=1)
plot.ts(kingtimeseriesdiff1)


acf(kingtimeseriesdiff1, lag.max=20, na.action=na.pass)             
acf(kingtimeseriesdiff1, lag.max=20, plot=FALSE, na.action=na.pass)


pacf(kingtimeseriesdiff1, lag.max=20)             
pacf(kingtimeseriesdiff1, lag.max=20, plot=FALSE)

## correlation after lag 1, partial correlation after lag 3 so good are arima(3,0), arima(0,1) and mixtures
## least number of parameters have arima(0,1) -> sma(1)


volcanodust <- scan("list8/dvi.dat", skip=1)
volcanodustseries <- ts(volcanodust,start=c(1500))
plot.ts(volcanodustseries)

acf(volcanodustseries, lag.max=20)  
acf(volcanodustseries, lag.max=20, plot=FALSE)  

pacf(volcanodustseries, lag.max=20)  
pacf(volcanodustseries, lag.max=20, plot=FALSE)  


### acf 2 pacf 1


### FORECASTING by ARIMA

kingstimeseriesarima <- arima(kingsts, order=c(0,1,1))
kingstimeseriesarima

kingstimeseriesforecast <- forecast(kingstimeseriesarima, h=5)
kingstimeseriesforecast

plot(kingstimeseriesforecast)

acf(kingstimeseriesforecast$residuals, lag.max=20)
Box.test(kingstimeseriesforecast$residuals, lag=20, type="Ljung-Box")

plot(kingstimeseriesforecast$residuals)        
plotForecastErrors(kingstimeseriesforecast$residuals)

## volcano 

volcanodustseriesarima <- arima(volcanodustseries, order=c(2,0,0))
volcanodustseriesarima

volcanodustseriesforecasts = forecast(volcanodustseriesarima, h=31)
volcanodustseriesforecasts

plot(volcanodustseriesforecasts)

acf(volcanodustseriesforecasts$residuals, lag.max=20)
Box.test(volcanodustseriesforecasts$residuals, lag=20, type="Ljung-Box")

plot.ts(volcanodustseriesforecasts$residuals)            
plotForecastErrors(volcanodustseriesforecasts$residuals)

mean(volcanodustseriesforecasts$residuals)
