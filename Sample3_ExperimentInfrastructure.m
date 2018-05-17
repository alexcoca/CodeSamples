%% Main function
% Load default Parameters
load inputStruct.mat
% Load results structure cofiguration
load resConfig.mat
% Change parameters
inputStruct.f = 'egg5';
inputStruct.dim = 5;
inputStruct.L = (-512)*ones(1,inputStruct.dim);
inputStruct.U = (+512)*ones(1,inputStruct.dim);
nruns = 50; % No of runs after which to evaluate 
afterN = 286; % No of fcn evals after which average std dev/mean of obj is des.
flag = 'avg'; %If set to flag, plots average objective change over nruns. Set to 'traces' to plot individual runs
traceIndex = [1,2,3]; %Select which runs to plot the objective change for 
inputStruct.nruns = nruns; %Set number of runs
% Configure experiment
[ResultsNew] = experiment(inputStruct,resConfig,'kEdg',0.09*ones(1,4),'kTR',0.03*ones(1,4),'gamma',0.2*ones(1,4),'lmain',[10,10,10,10],'lmainP',[4,4,4,4],'NTL',5*ones(1,4),'linner',[6,8,14,16],'linnerP',[3,4,7,8]);
% Post process results
[fbest2,fbestloc2,fworst2,fworstloc2,stdevs2,means2,avg_counts2,avg_tpr,~,avgAfterN,stdAfterN,xmin] = postProcess2(nruns,traceIndex,flag,afterN,inputStruct.dim,ResultsNew);
%% Function that automates experiment execution for DTSo
function [Results] = experiment(inputStruct,resStruct,varargin)
% Generate names for experiments & detect which variables change
[names,varNames,varValues] = generateNames(varargin);
% Create results container - a standard structure with experiments as field
% names
Results = struct;
for index = 1:length(names)
    Results.(names{index})=[];
end
%Run experiment for a certain parameter configuration(outer loop)
for j=  1:size(varValues,2)
    %But set loop through variable configurations
    for i = 1:length(varNames)
        inputStruct.(varNames{i}) = varValues(i,j);
    end
    out = runParamSetting(inputStruct,resStruct);
    Results.(names{j}) = out;
end
end
%% %% Helper function: generates experiements names
function [names,varNames,varValues] = generateNames(varargin)
varNames ={};
varValues = [];
j=1;
for i = 1:length(varargin{1})
    if ischar(varargin{1}{i}) == 1
        varNames{end+1} = varargin{1}{i};
    else
        varValues(j,:) = varargin{1}{i};
        j = j+1;
    end
end
components = cell(length(varNames),size(varValues,2));
for j = 1:length(varNames)
    for index = 1:size(varValues,2)
        components{j,index} = strcat(varNames{j},num2str(varValues(j,index)));
    end
end
names = cell(1,size(varValues,2));
for j=1:size(components,2)
    names{j} = strcat(components{:,j});
end

for j=1:length(names)
    names{j} = strrep(names{j},".","");
end
end
%%
%% Helper function: run DTS with a specific configuration and collect res.
function [Results] = runParamSetting(inputStruct,Results)
for run=1:inputStruct.nruns
    [XMin,FMin,FCount,VRL,frVRL,fTL,TL,recOUT,fevol,lhist,histout,runtime,explorationRunTimes,diversificationRunTimes,NeMeStatus,fhistory,Shist,TLhist,VRLhist] = DTSps(inputStruct.f,inputStruct.U,inputStruct.L,inputStruct.nIterDMax,run,inputStruct.gamma,inputStruct.Nbest,inputStruct.lmain,inputStruct.lmainP,inputStruct.linner,inputStruct.linnerP,inputStruct.kTR,inputStruct.kSTR,inputStruct.kEdg,inputStruct.NTL,inputStruct.frNTLbar,inputStruct.kVR,inputStruct.cflag);
    Results(run).XMin = XMin;
    Results(run).FMin = FMin;
    Results(run).FCount = FCount;
    Results(run).VRL = VRL;
    Results(run).frVRL = frVRL;
    Results(run).fTL = fTL;
    Results(run).TL = TL;
    Results(run).recOUT = recOUT;
    Results(run).FEvol = fevol;
    Results(run).lhist = lhist;
    Results(run).histout = histout;
    Results(run).runtime = runtime;
    Results(run).explorationRunTimes = explorationRunTimes;
    Results(run).diversificationRunTimes = diversificationRunTimes;
    Results(run).NeMeStatus = NeMeStatus;
    Results(run).fhistory = fhistory;
    Results(run).Shist = Shist;
    Results(run).TLhist = TLhist;
    Results(run).VRLhist = VRLhist;
end
end
% Note: DTS function not written by myself so not included.

