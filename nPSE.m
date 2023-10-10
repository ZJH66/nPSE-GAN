function [NC, TP]=nPSE()
TC = load('\sim1.txt');
[NC, TP] = getA(TC(1:1000,:),5,4,200,' ')
end
