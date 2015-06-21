fname = 'a104s.mat'
fname = 'a161l.mat'
load(fname)

figure
start = 1
final = 10000

final = length(val(1,:))
% start = final - 10000
subplot(311)
plot(val(1,start:final))
subplot(312)
plot(val(2,start:final))
subplot(313)
plot(val(3,start:final))