function [walls] = fayyadIrani (points, classes, intervals)

% INPUT             
% 	(1) points - the 1-dimensional values of each instance
%   (2) classes - class numbers {0,1} of all points, namely class labels of all points.
%   (3) intervals - how much you want to break up each smaller section. OPTIONAL. The default is 40. 
%
% OUTPUT            
% 	(1) walls - vector containing all "walls"
% NOTES
%   This is a pretty neat implementation of information theory.  In my empirical
%   observation, this is by far the fastest discretizing metric I've come across.
%   In addition, it seems to be quite accurate.
% CITATIONS
%   code implements Fayyad and Irani's discretization algorithms as described 
%   in Kohavi, Dougherty - 'Supervised and Unsupervised Discretization of
%   Continuous Features' and 'Li, Wong' - Emerging Patterns and Gene Expression 
%   data =>
%   http://hc.ims.u-tokyo.ac.jp/JSBi/journal/GIW01/GIW01F01/GIW01F01.html.
%
% Lawrence David - 2003.  lad2002@columbia.edu
% Tom Macura - 2005. tm289@cam.ac.uk (modified for inclusion in OME)

% warning off MATLAB:colon:operandsNotRealScalar;         % shutup

if (nargin < 3)
	intervals = 40; % 40 - original
end

walls   = [];
points  = double(points);                   % sometimes, data comes in as 'single'
classes = double(classes);

% Erect walls between min and max to divide distance into interval intervals
bin_walls = min(points):(max(points)-min(points))/intervals:max(points); % A vector, begin form min(points), end at max(points), the gap is '(max(points)-min(points))/intervals'

if ~isempty(bin_walls) % stop if you split into space with no points
	bin_num = length(bin_walls); % 
	N       = length(points);
	
	% compute class entropy and number of unique classes
	[s_ent class_number] = class_entropy(classes); % -sum(s_temp.*log2(s_temp))
	class_info_ent  = [];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% looking for best way to split points into two parts                   
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
	for i = 1:bin_num
	
		% Divid samles of a feature into two parts 
        temp = (points > bin_walls(i)) + 1;        % logical value plus a numeric value becomes a numeric value；simple discretization;two cases:<= & >
        
		% Corrsponding labels of each part
		s1 = classes(find(temp==1)); % 'find(temp==1)' denotes the index of elements which are not larger than bin_walls(i)
        s2 = classes(find(temp==2)); % 'find(temp==2)' denotes the index of elements which are larger than bin_walls(i)
		
		% Number of elements in each part
        s1_size = length(s1);
        s2_size = length(s2);
		
		% Clss entropy
        [s1_ent(i) s1_k(i)] = class_entropy(s1);   % get class entropies by calling self-defined function
        [s2_ent(i) s2_k(i)] = class_entropy(s2);   % get class entropies
		
        class_info_ent(i) = (s1_size/N)*s1_ent(i) + (s2_size/N)*s2_ent(i);  % want to minimize this baby; Definition by Fayyad and Irani's discretization algorithms
	end
	
	[low_score lsp] = min(class_info_ent);          % where was entropy minimized? return the minimum and its position
	
	p1 = points(find(points < bin_walls(lsp)));     % number games . . . 
	p2 = points(find(points > bin_walls(lsp)));
	
	c1 = classes(find(points < bin_walls(lsp)));
	c2 = classes(find(points > bin_walls(lsp)));
	
	gain = s_ent-class_info_ent(lsp);               % do we have enough information gain?
	
	deltaATS = log2(3^class_number - 2) - (class_number*s_ent - s1_k(lsp)*s1_ent(lsp) -s2_k(lsp)*s2_ent(lsp));
	right_side = log2(N-1)/N + deltaATS/N; % 
	
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% descend recursively                                           %
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
	if ~(gain < right_side)
		%
		% FIXME intervals is hard-coded to 25. WTF?
		%
		intervals = 25;
        walls = [walls bin_walls(lsp) fayyadIrani(p1,c1,intervals) ...
            fayyadIrani(p2,c2,intervals)]; 
	else
        walls = [];
	end
else
    walls = [];         % if you're having no luck, just give up
end

end