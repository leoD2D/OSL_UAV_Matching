import numpy as np
import copy


class unmannedAerialVehicle:

    def __init__(self, index, mode):
        self.mode = mode
        self.index = index
        self.xPositionUAV = int(np.random.uniform(0,100))
        self.yPositionUAV = int(np.random.uniform(0,100))
        self.cpuFrequencyMaxUAV = int(np.random.uniform(10, 30) * 10**9)   #30 GHz
        self.bandwidthMaxUAV = int(np.random.uniform(1 ,2 )* 10**6)     # 2 MHz
        self.ownSensingTaskSizeUAV =int(np.random.uniform(1 ,3) * 10**6)    # 2 MHZ
        self.ownSensingTaskComplexityUAV =int(np.random.uniform(2000, 10000))
        self.penalty = -1
        self.reward = 1
        self.beta = {}
        self.selectedBeta = {}

        self.receivedProposals = 0
        self.proposingGDs = []
        self.preferenceOfGDs = {}

        self.utilityUAV = 0
        self.listOfAcceptedGDs = []
        self.numberOfAcceptedGDs = 0
        self.connectedGDsInTimestep = []
        self.utilityInTimestepUAV = 0
        self.utilityPerTimestepUAV = {}

        self.maxEpsilon = 1
        self.rng = np.random.default_rng()
        self.timestep = 1

    def marginalUtility(self, listOfGDs):
        listOfGDs = copy.deepcopy(listOfGDs)
        if not listOfGDs:
            marginalUtilityForGDs = 0
        else:
            cpuFreq = self.cpuFrequencyMaxUAV / (len(listOfGDs) + 1)
            bandwidth = self.bandwidthMaxUAV / len(listOfGDs)
            t_extra = self.ownSensingTaskSizeUAV * self.ownSensingTaskComplexityUAV * (
                        1 / cpuFreq - 1 / self.cpuFrequencyMaxUAV)
            t_saved = 0
            for GD in listOfGDs:
                t_local = GD.timeLocalGD
                channelRate = bandwidth * np.log(1 + ((GD.channelGain * GD.transPower) / (bandwidth * GD.noise)))
                # print('channel rate:', channelRate)
                t_comm = GD.taskSizeGD / channelRate
                t_comp = GD.taskSizeGD * GD.taskComplexityGD / cpuFreq
                t_saved += (t_local - t_comp - t_comm)

            marginalUtilityForGDs = self.reward * t_saved + self.penalty * t_extra

            if marginalUtilityForGDs is None or marginalUtilityForGDs.size == 0:
                marginalUtilityForGDs = 0
            # print(f"The marginal Utility is real {marginalUtilityForGDs} with t_saved = {t_saved} and t_extra = {t_extra}")

        return marginalUtilityForGDs

    def calculateMaximalUtility(self, proposingGDs, preferenceOfGDs):
        proposingGDs = copy.deepcopy(proposingGDs)
        preferenceOfGDs = copy.deepcopy(preferenceOfGDs)
        sorted_prefListOfGDs = []
        l_minuseins =[]
        l = []

        for ii in preferenceOfGDs.keys():
            for GD in proposingGDs:
                if ii == GD.index:
                    sorted_prefListOfGDs.append(GD)

        for GD in sorted_prefListOfGDs:
            l.append(GD)
            if self.marginalUtility(l)> self.marginalUtility(l_minuseins):
                self.listOfAcceptedGDs.append(GD.index)
                self.numberOfAcceptedGDs = len(l)
                self.utilityUAV = self.marginalUtility(l)
            l_minuseins = l

        #print(f"The accepted GDs by {self.index} are: {self.listOfAcceptedGDs}")

    def chooseAcceptedGDs(self, listOfGDs, listOfProposingGDs):
        listOfGDs = copy.deepcopy(listOfGDs)
        listOfProposingGDs = copy.deepcopy(listOfProposingGDs)

        for GD in listOfGDs:
            if GD.index in listOfProposingGDs:
                self.proposingGDs.append(GD)

        for GD in self.proposingGDs:
            listForUtility =[GD]
            self.preferenceOfGDs.update({GD.index : self.marginalUtility(listForUtility)})
        sorted_listpreferenceOfGDs = sorted(self.preferenceOfGDs.items(), key=lambda item: item[1], reverse=True)
        self.preferenceOfGDs = dict(sorted_listpreferenceOfGDs)
        #print(self.index, "has the preferences of the proposing GDs:", self.preferenceOfGDs)

        self.calculateMaximalUtility(self.proposingGDs, self.preferenceOfGDs)


    def updateUtilityOfUAVs(self, listOfGDs, timestep):
        listOfGDs = copy.deepcopy(listOfGDs)
        if not self.listOfAcceptedGDs:
            self.utilityInTimestepUAV = 0
        else:
            t_saved = 0
            cpuFreq = self.cpuFrequencyMaxUAV / (len(self.listOfAcceptedGDs) + 1)
            bandwidth = self.bandwidthMaxUAV / len(self.listOfAcceptedGDs)
            t_extra = self.ownSensingTaskSizeUAV * self.ownSensingTaskComplexityUAV * (1 / cpuFreq - 1 / self.cpuFrequencyMaxUAV)
            for GD in listOfGDs:
                if GD.index in self.listOfAcceptedGDs:
                    t_local = GD.timeLocalGD
                    channelRate = bandwidth * np.log(1 + ((GD.channelGain * GD.transPower) / (bandwidth * GD.noise)))
                    t_comm = GD.taskSizeGD / channelRate
                    t_comp = GD.taskSizeGD * GD.taskComplexityGD / cpuFreq
                    t_saved += (t_local - t_comp - t_comm)

            self.utilityInTimestepUAV = self.reward * t_saved + self.penalty * t_extra
        self.utilityPerTimestepUAV.update({timestep : self.utilityInTimestepUAV})
        return self.utilityInTimestepUAV


    def unmatchGDfromUAV(self):
        self.receivedProposals = 0
        self.proposingGDs = []
        self.preferenceOfGDs = {}

        self.utilityUAV = 0
        self.listOfAcceptedGDs = []
        self.numberOfAcceptedGDs = 0
        self.connectedGDsInTimestep = []
        self.utilityInTimestepUAV = 0
        self.selectedBeta ={}

















