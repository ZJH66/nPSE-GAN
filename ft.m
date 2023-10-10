function [nc,tp,X,c]=ft(n,Alen,A)
y = zeros(n*(n-1),Alen);
count = 1; 
for i=1:n
    for j=1:n
        if i~=j
            for k=1:Alen
                y(count,k) = A(i,j,k);
            end
            count = count+1;
        end
    end
end
X=y';
[c, num_clust]= hc(X);
nc=num_clust(1,end);
km=c(:,end); %从 c 中提取最后一列，即聚类结果，赋给 km。
tp=[];%簇的边界的索引
for i=1:Alen-1
    if km(i)~=km(i+1)
        tp = [tp,i];
        if i>1 && km(i)~=km(i-1)
            km(i)=km(i-1);
            tp(end)=[];
            tp(end)=tp(end)+1;
            if km(i)==km(i+1)
                tp(end)=[];
            end
        end
    end
end
% converging the transition 
tp = [tp,Alen];
value = 20; 
i=2;
if size(tp,2)>1
while tp(i)~=Alen
    interval = tp(i)-tp(i-1);
    if interval<=value && km(tp(i-1))==km(tp(i)+1)
        km((tp(i-1)+1):tp(i))= km(tp(i-1));
        tp(i-1)=[];
        i=i-1;
        tp(i)=[];
        i=i-1;
    end
    i=i+1;
    if i==1
        if tp(i)~=Alen
            i=2;
        else
            break;
        end
    end
end
end
% disp(km)
%rectifying the transition
i=2;
if size(tp,2)>1
    tmp = nc;
    while i~=tmp
        % 计算 tanh[(|sm| + |c_jclu|) * n / 2*T] 的双曲正切函数值
        eta = tanh((tp(i)-tp(i-1)+ sum(km(tp(i-1):tp(i)))) * nc / (2*Alen));
        disp(eta)
        % 根据条件判断是否消除状态
        if eta < 0.3% occurrence probability
            nc = nc- 1;
        %  消除状态的相应处理           
        %  对 tp 和 km 进行相应的修改，例如：
            km((tp(i-1)+1):tp(i))= km(tp(i-1));
            tp(i-1)=[];
        end
    i=i+1;
    end
end
end