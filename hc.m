function [c, num_clust]= hc(data)
   min_sim=inf;
   [Affinity_,  orig_dist, ~]= rank(data,[]);
  [Group_] = clust(Affinity_, [],inf);  
  [c,num_clust, mat]=merge([],Group_,data);
      if ~isempty(orig_dist)
       min_sim=  double(max(orig_dist(Affinity_>0)));
    end
    exit_clust=inf;
    c_=c;
k=2;
while exit_clust>1
    [Affinity_,  orig_dist,~]= rank(mat,[]);      
    [u] = clust(Affinity_, double(orig_dist),min_sim);
    [c_,num_clust_curr, mat]=merge(c_, u, data);     
    num_clust =[num_clust, num_clust_curr]; 
    c = [c, c_];        
   exit_clust=num_clust(end-1)-num_clust_curr;  
     if num_clust_curr==1 || exit_clust<1
         num_clust=num_clust(1:end-1);
         c=c(:,1:end-1);
         exit_clust=0;
        break
     end
    k=k+1;   
end
end

function [A, orig_dist,min_sim]= rank(mat, initial_rank)
s=size(mat,1);  
if ~isempty(initial_rank)
        orig_dist=[]; min_sim=inf;
else
 orig_dist = pdist2(mat,mat,'cosine'); 

%  orig_dist = shd(mat,s);
 orig_dist(logical(speye(size(orig_dist))))=inf;
 [d,initial_rank]=min(orig_dist,[],2);
 min_sim=max(d);
end
  A=sparse([1:s],initial_rank,1,s,s);
  A= A + sparse([1:s],[1:s],1,s,s);
  A= (A*A');
  A(logical(speye(size(A))))=0; 
  A=spones(A);
end

function [u]= clust(A, orig_dist,min_sim)
 if min_sim~=inf
 ind=find((orig_dist.*A)> min_sim) ;
 A(ind)=0; 
 end
 G_d=digraph(A, 'OmitSelfLoops');
 u = (conncomp(G_d,'Type','weak'));
end

function [c,num_clust, mat]=merge(c,u,data)
    u_ =ind2vec(u); num_clust=size(u_,1);           
    if ~isempty(c)
     c=getC(c,u');
     else
        c=u';
    end   
%       if num_clust<=5e6
%           [mat] =mean(data,c); 
%       else 
         [~,ic,~] = unique(c,'last');
         mat= data(ic,:);
%       end
   end      
 function G=getC(G,u)  
    [~,~,ig]=unique(G);
    G=u(ig);     
 end
 











