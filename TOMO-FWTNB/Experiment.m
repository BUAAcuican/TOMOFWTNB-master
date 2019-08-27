%% An example of TOMOFWTNB 
% Add path
str = pwd;
ind = strfind(pwd,'\');
if (~strcmp('TOMO-FWTNB',str(ind(end)+1:end))) % Judge current path is right or not
    msgbox('Please ensure that current path is ''...\\TOMO-FWTNB''','Error');
    return;
else
    addpath(str(1:ind(end)-1))
end

dataNames = {{'camel-1.0','ivy-1.4'}, {'jedit-4.0', 'ivy-1.4'}};

% Set saving path of emperiment results
filePath = 'D:\Experiments\';
if ~exist(filePath,'dir')
    mkdir(filePath);
end

dataPath = [pwd, '\'];

% Experiment Settings 
runtimes = 30; % the number of runnings of prediction model
percent_tt = 0.9;% the percentage of training data in source data

perfNames = {'PD','PF','Precision','F1','AUC','Accuracy','G-Measure','MCC'};
modelNames = {'TOMO-FWTNB'};


expNames = [];
for i=1:numel(dataNames{1})
    expNames{i} = [dataNames{1}{i},'_',dataNames{2}{i}];
end
sources = [];
targets = [];

perfs = cell(1,numel(dataNames{1}));

import weka.filters.*; 
import weka.*;

for d=1:numel(dataNames{1}) % Each dataset
    
    disp(['Data: ',num2str(d),' / ',num2str(numel(dataNames{1}))]);
       
    % Load data    
    file1 = java.io.File([dataPath,dataNames{1}{d},'.arff']);  % create a Java File object (arff file is just a text file)
    loader = weka.core.converters.ArffLoader;  % create an ArffLoader object
    loader.setFile(file1);  % using ArffLoader to load data in file .arff
    insts = loader.getDataSet; % get an Instances object
    insts.setClassIndex(insts.numAttributes()-1); %  set the index of class label
    [sources,featureNames,targetNDX,stringVals,relationName1] = weka2matlab(insts,[]); %{false,true}-->{0,1}
    sources = [sources(:, 1:end-1), double(sources(:, end)>0)]; % If defects(i) > 0, then defects(i) = 1, otherwise defects(i) = 0.
    
    file2 = java.io.File([dataPath,dataNames{2}{d},'.arff']);  % create a Java File object (arff file is just a text file)
    loader = weka.core.converters.ArffLoader;  % create an ArffLoader object
    loader.setFile(file2);  % using ArffLoader to load data in file .arff
    insts = loader.getDataSet; % get an Instances object
    insts.setClassIndex(insts.numAttributes()-1); %  set the index of class label
    [targets,featureNames,targetNDX,stringVals,relationName2] = weka2matlab(insts,[]); %{false,true}-->{0,1}
    targets = [targets(:, 1:end-1), double(targets(:, end)>0)];
    
    % Remove duplicated instances
    sources = unique(sources,'rows');
    targets = unique(targets,'rows');
    
    % Remove instances having missing values
    [idx_r idx_c] = find(isnan(sources));
    sources(unique(idx_r),:) = [];
    [idx_r idx_c] = find(isnan(targets));
    targets(unique(idx_r),:) = [];
    

    % Shuffle the instances : only when runtimes > 1 
    if runtimes > 1
%         rand('state',1);
        sources = sources(randperm(size(sources,1),size(sources,1)),:); % Disrupt the order of rows, which is beneficial to the following random resampling.
        targets = targets(randperm(size(targets,1),size(targets,1)),:);
    end
    
 
    % Predefine (this can reduce memory usage)
    PD_fw=zeros(runtimes,1);PF_fw=zeros(runtimes,1);Precision_fw=zeros(runtimes,1);F1_fw=zeros(runtimes,1);AUC_fw=zeros(runtimes,1);Accuracy_fw=zeros(runtimes,1);G_measure_fw=zeros(runtimes,1);MCC_fw=zeros(runtimes,1);
    
    % targetsCopy = targets;
    sourcesCopy = sources;
    
    
    for i=1:runtimes
        disp(['runtimes:',num2str(i),' / ',num2str(runtimes)]);
        
        idx = randperm(size(sourcesCopy,1),round(percent_tt*size(sourcesCopy,1)));  %
        trainData = sourcesCopy(idx,:);
        while numel(unique(trainData(:,end)))==1
            idx = randperm(size(sourcesCopy,1),round(percent_tt*size(sourcesCopy,1)));
            trainData = sourcesCopy(idx,:);
        end
        sources = trainData;
        
        
        %% Proposed FWTNB
        disp('FWTNB ...');
        source = sources;
        target = targets;
        [ PD_fw(i,:),PF_fw(i,:),Precision_fw(i,:),F1_fw(i,:),AUC_fw(i,:),Accuracy_fw(i,:),G_measure_fw(i,:),MCC_fw(i,:) ] = TOMOFWTNB(source, target);
        
    end % End of runs
    perfs{d} = {[PD_fw,PF_fw,Precision_fw,F1_fw,AUC_fw,Accuracy_fw,G_measure_fw,MCC_fw]};

    save([filePath,'\perfs.mat'],'perfs'); 
end % End of datsets

a = 1;
