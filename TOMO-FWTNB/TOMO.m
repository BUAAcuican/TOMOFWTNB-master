function [ new_dataSet ] = TOMO( dataSet,target_x, ratio, lamda )
%TOMO Summary of this function goes here
%   Detailed explanation goes here
% INPUTS:
%   (1) dataSet - A mat, each row is an instance, the last column is the class
%   labels {0,1} where 0 denotes non-defec-proneness (i.e., the minority
%   class) and 1 denotes defect-proneness (i.e., the majority class). All
%   features (metrics) are numeric.
%   (2) target_x - each column is a feature
%   (3) ratio   - A number with range of [0,1], which denotes the ideal
%   defect ratio of defect-prone modules to non-defect-prone ones.
% OUTPUTS:
%   newdataSet  - A mat with ideal defect ratio, which has same number of columns as dataSet. 

label = dataSet(:,end);
if(length(label==1)>length(label==0))
    error('label 1 must denotes the minority class');
end

if nargin < 4
    lamda = 0.4; % By default
end

% Number of positive instances (n_p), number of negative insatnces 
n_p = length(find(label==1));
n_n = length(find(label==0));

% Number of new samples needed to be generated
n0 = floor(n_n*ratio) - n_p;

% positive/negative subset
dataP = dataSet(find(label==1),:);
dataN = dataSet(find(label==0),:);

% 
dataP_x = dataP(:,1:(end-1));

% Distance between instances
distM = dist(dataP_x'); % dist(P) takes an RxQ matrix P of Q R-element column vectors, and returns a QxQ matrix of the distances between each of the Q vectors.

%% Case1

%% Case2
% cluster centroid of potential minority class in target data
% K-means
[idx,C] = kmeans(target_x,2); %

minoC = [];
if(unique(C)==2) % two cluster 
    if (sum(idx==1)>sum(idx==2))
        minoC = C(2,:);
    elseif(sum(idx==1)<sum(idx==2))
        minoC = C(1,:);
    else
        if(norm(C(1,:))>norm(C(2,:)))
            minoC = C(1,:);
        else
            minoC = C(2,:);
        end
    end
else % just one cluster
    minoC = C(1,:);
end

% minoC = mean(target_x,1);

% distance between above centroil and samples in dataP
D = pdist([minoC;dataP_x]);
D = D(1:n_p);
[value,idx] = sort(D); % ascending order

dataP_x_asc = dataP_x(idx,:);
dataP_x = dataP_x_asc;
distSS = dist(dataP_x_asc'); % For an RxQ matrix P, dist(P) returns a QxQ matrix of the distances between each of the Q vectors.

temp = pdist([minoC;dataP_x_asc]);
distTS = repmat(temp(1:n_p),n_p,1);


% Normalization
distSS_Norm = distSS./repmat(sum(distSS,2),1,n_p);
distTS_Norm = distTS./repmat(sum(distTS,2),1,n_p);


%
%lamda = 0.4; % By default
distHybrid = lamda*distSS_Norm +(1-lamda)*distTS_Norm;


% Index of neighbors of each instance in dataP_x_asc
nearNeigIndex = zeros(size(distHybrid,1),size(distHybrid,1)-1);
for i=1:size(distHybrid,1)
    [val, ord] = sort(distHybrid(i,:)); % smallest->biggest, the value of 'ord' denotes the line number of instances in dataP_x_asc.
    ord(find(ord==i))=[];
    nearNeigIndex(i,:) = ord;   
end

%%


% Initialize new data
new_x = zeros(n0,size(dataP_x,2));

% Number of being allocated neighbors for each positive instance
k = n0 / n_p;

if k<1
    % (1) Randomly select n0 original positive samples
%     ind = randperm(n_p,n0);
    
    ind = 1:n0;
    
    % (2) Generate a new sample for each of above positive samples
    for i=1:length(ind)
        temp = dataP_x(ind(i),:) + rand * (dataP_x(nearNeigIndex(ind(i),1),:) - dataP_x(ind(i),:));
        new_x(i,:) = temp;
    end
    
elseif (k>=1)&&(k<=(n_p-1))

    % (1) First generate floor(k) new samples for each original positive sample
    j0 = 0;
    for i=1:size(dataP,1)
        for j=1:floor(k)
            j0 = j0 + 1;
            temp = dataP_x(i,:) + rand * (dataP_x(nearNeigIndex(i,j),:) - dataP_x(i,:));
            new_x(j0,:) = temp;
        end
    end
    
    % (2) Randomly select (n0-(floor(k)*size(dataP,1))) original positive samples
%     ind = randperm(size(dataP,1), n0 - (floor(k)*size(dataP,1)));
    ind = 1:(n0 - (floor(k)*size(dataP,1)));
    
    % Generate a new sampel for each of above origianl positive samples
    for i=1:length(ind)
        j0 = j0 + 1;
        temp = dataP_x(ind(i),:) + rand * (dataP_x(nearNeigIndex(ind(i),floor(k)+1),:) - dataP_x(ind(i),:));
        new_x(j0,:) = temp;
    end   
else

	% 
	k0 = floor(k) / (n_p-1); 
	
	% (1) First generate floor(k0)*(n_p-1) new samples for each original positive sample
	j0 = 0;
	for i0=1:floor(k0)
		for i=1:n_p
			for j=1:n_p-1
				j0 = j0 + 1;
				temp = dataP_x(i,:) + rand * (dataP_x(nearNeigIndex(i,j),:) - dataP_x(i,:));
				new_x(j0,:) = temp;
			end
		end
	end
	
	% (2) Secondely generate some new samples for each original positive sampels
	for i=1:n_p
		for j=1:(floor(k)-(floor(k0))*(n_p-1))
            j0 = j0 + 1;
			temp = dataP_x(i,:) + rand * (dataP_x(nearNeigIndex(i,j),:) - dataP_x(i,:));
            new_x(j0,:) = temp;
		end
	end
	
	% (3) Randomly select (n0-(floor(k)*size(dataP,1))) original positive samples
% 	ind = randperm(size(dataP,1), n0 - (floor(k)*size(dataP,1)));
	ind = 1:(n0 - (floor(k)*size(dataP,1)));
    
	% Generate a new sample for each of above original positive samples
    for i=1:length(ind)
        j0 = j0 + 1;
        temp = dataP_x(ind(i),:) + rand * (dataP_x(nearNeigIndex(ind(i),floor(k)-(floor(k0))*(n_p-1)+1),:) - dataP_x(ind(i),:));
        new_x(j0,:) = temp;
    end
	
end

% Add class labels (i.e., a column vector of 1 where 1 denotes defect-prone)
new_dataSet = [new_x, ones(size(new_x,1),1)];

% Combine old and new data
new_dataSet = [dataSet;new_dataSet];

end

