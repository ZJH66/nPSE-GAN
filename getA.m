function [NC, TP]= getA(TC,n,subj,sublen,dataname)
A=dekf(TC',1);
Alen = size(A,3);
for i=1:Alen
    for j=1:n
        for k=1:n
            if A(j,k,i)<=0
                A(j,k,i)=0;
            else
                A(j,k,i)=1; % +0.8*(fix(i/(sublen+1))+1);
            end
        end
    end
end
An=zeros(n,n,sublen);
for i=1:sublen
    B=zeros(n);
    for j=1:subj
        B=B+A(:,:,(j-1)*sublen+i); 
    end
    B=B/subj;
    for k=1:n
        for l=1:n
            if B(k,l)>=0.6 && k~=l
                B(k,l)=1;
            else
                B(k,l)=0;
            end
        end
    end
    An(:,:,i)=B;
end

NC=[];
TP=[];
[nc,tp,X,c]=ft(n,sublen,An);
NC=[NC,nc];
TP=[TP,tp(1:end-1)];
end