import numpy as np
import copy
import random

class groundDevices:
    def __init__(self, index, mode):
        self.mode = mode
        self.index = index
        self.xPositionGD = int(np.random.uniform(0,100))
        self.yPositionGD = int(np.random.uniform(0,100))
        self.cpuFrequencyGD =int(np.random.uniform(1,2)) *10**(9)
        self.taskSizeGD =int(np.random.uniform(1,5)) * 10**6    #1 Mbit
        self.taskComplexityGD =int(np.random.uniform(2000,10000))
        self.transPower = 0.1   # 100 mW
        self.noise = 10.0 ** (-90 / 10.0)*0.001       # -90 dBm

        self.alpha_ssp = 0.01         #step-size parameter alpha >= 0
        self.possibleUAVs = []
        self.probFailedConnection = {}
        self.preferenceOfUAVsToPropose = {}
        self.probabilityOfUAVsToPropose = {}
        self.averageRewardGD = 0

        self.distance = 0
        self.channelGain = 0

        self.timeLocalGD = 0
        self.timeCompGD = 0
        self.timeCommGD = 0

        self.chosenUAVtoPropose = None

        self.rewardInTimestepGD = 0
        self.utilityPerTimestepGD = {}

        self.y = 0





    def initializeGD(self,listOfUAVs):
        listOfUAVs = copy.deepcopy(listOfUAVs)
        self.possibleUAVs.extend(UAV.index for UAV in listOfUAVs)
        # print(self.possibleUAVs)
        self.preferenceOfUAVsToPropose = {'No Proposal':0}
        self.preferenceOfUAVsToPropose.update({UAV.index: 0 for UAV in listOfUAVs})
        #print(self.preferenceOfUAVsToPropose)
        self.probabilityOfUAVsToPropose = {'No Proposal': 0}
        self.probabilityOfUAVsToPropose.update({UAV.index: 0 for UAV in listOfUAVs})
        #print(self.probabilityOfUAVsToPropose)

    def chooseUAVtoPropose(self, listOfUAVs):
        listOfUAVs = copy.deepcopy(listOfUAVs)

        if self.mode == 'oneSidedLearning':
            # Convert dictionary values to a NumPy array
            preference_values = np.array(list(self.preferenceOfUAVsToPropose.values()))

            # Calculate exponentials of the preferences
            exp_preferences = np.exp(preference_values)
            self.probabilityOfUAVsToPropose['No Proposal'] = np.exp(
                self.preferenceOfUAVsToPropose['No Proposal']) / np.sum(exp_preferences)
            for UAV in listOfUAVs:
                self.probabilityOfUAVsToPropose[UAV.index] = np.exp(self.preferenceOfUAVsToPropose[UAV.index]) / np.sum(
                    exp_preferences)
            probabilities = list(self.probabilityOfUAVsToPropose.values())
            # print(f"the Probabilities are: {probabilities}")
            self.chosenUAVtoPropose = np.random.choice(list(self.probabilityOfUAVsToPropose.keys()), p=probabilities)
            # print(self.chosenUAVtoPropose)
            self.provideInformationToUAV(self.chosenUAVtoPropose, listOfUAVs)
            return self.chosenUAVtoPropose

        if self.mode == 'UAVsOnlyMatching':
            self.chosenUAVtoPropose = random.choice(self.possibleUAVs)
            self.provideInformationToUAV(self.chosenUAVtoPropose, listOfUAVs)
            return self.chosenUAVtoPropose

        if self.mode == 'ChannelGainBasedMatching':
            channelGains = {}
            for UAV in listOfUAVs:
                distanceToUAV = np.sqrt(
                    (self.xPositionGD - UAV.xPositionUAV) ** 2 + (self.yPositionGD - UAV.yPositionUAV) ** 2)
                if distanceToUAV < 2:
                    distanceToUAV = 2
                channelGainToUAV = distanceToUAV ** (-4)
                channelGains[UAV.index] = channelGainToUAV

            self.chosenUAVtoPropose = max(channelGains, key=channelGains.get)
            self.provideInformationToUAV(self.chosenUAVtoPropose, listOfUAVs)
            return self.chosenUAVtoPropose

    def provideInformationToUAV(self, chosenUAV, listOfUAVs):
        chosenUAV = copy.deepcopy(chosenUAV)
        listOfUAVs = copy.deepcopy(listOfUAVs)
        #self.taskSizeGD = int(np.random.uniform(1, 5)) * 10 ** 6  # 1 Mbit
        #self.taskComplexityGD = int(np.random.uniform(2000, 10000))
        self.timeLocalGD = self.taskSizeGD * self.taskComplexityGD / self.cpuFrequencyGD
        #print(f"the local Time of {self.index} is: {self.timeLocalGD}")
        for UAV in listOfUAVs:
            if UAV.index == chosenUAV:
                self.distance = np.sqrt((self.xPositionGD - UAV.xPositionUAV)**2 + (self.yPositionGD - UAV.yPositionUAV)**2)
                if self.distance < 2:
                    self.distance = 2
                self.channelGain = self.distance**(-4)


    def updateUtilityOfGD(self, listOfUAVs, timestep):
        listOfUAVs = copy.deepcopy(listOfUAVs)
        counterNotAccepted = 0

        if self.chosenUAVtoPropose == 'No Proposal':            #Case 1: The GD did not proposed to any UAV and calculates the task locally
            self.rewardInTimestepGD = 0
            self.preferenceOfUAVsToPropose['No Proposal'] = self.preferenceOfUAVsToPropose['No Proposal'] + self.alpha_ssp * (self.rewardInTimestepGD - self.averageRewardGD) * (1 - self.probabilityOfUAVsToPropose['No Proposal'])
            for UAV in listOfUAVs:
                self.preferenceOfUAVsToPropose[UAV.index] = self.preferenceOfUAVsToPropose[UAV.index] - self.alpha_ssp * (self.rewardInTimestepGD - self.averageRewardGD) * (self.probabilityOfUAVsToPropose[UAV.index])
            self.averageRewardGD = self.averageRewardGD + ((self.rewardInTimestepGD - self.averageRewardGD) / timestep)

        else:
            for UAV in listOfUAVs:
                if self.index in UAV.listOfAcceptedGDs:         #Case 2: The GD proposed to a UAV and got accepted
                    cpuFreq = UAV.cpuFrequencyMaxUAV / len(UAV.listOfAcceptedGDs)
                    bandwidth = UAV.bandwidthMaxUAV / len(UAV.listOfAcceptedGDs)
                    channelRate = bandwidth * np.log(1 + ((self.channelGain * self.transPower) / (bandwidth * self.noise)))
                    self.timeCommGD += self.taskSizeGD / channelRate
                    self.timeCompGD += self.taskSizeGD * self.taskComplexityGD / cpuFreq
                    self.rewardInTimestepGD = self.timeLocalGD - self.timeCompGD - self.timeCommGD

                    self.preferenceOfUAVsToPropose[UAV.index] = self.preferenceOfUAVsToPropose[UAV.index] + self.alpha_ssp * (self.rewardInTimestepGD - self.averageRewardGD) * (1 - self.probabilityOfUAVsToPropose[UAV.index])
                    self.preferenceOfUAVsToPropose['No Proposal'] = self.preferenceOfUAVsToPropose['No Proposal'] - self.alpha_ssp * (self.rewardInTimestepGD - self.averageRewardGD) * (self.probabilityOfUAVsToPropose['No Proposal'])
                    for ii in listOfUAVs:
                        if ii.index != UAV.index :
                            self.preferenceOfUAVsToPropose[ii.index] = self.preferenceOfUAVsToPropose[ii.index] - self.alpha_ssp * (self.rewardInTimestepGD - self.averageRewardGD) * (self.probabilityOfUAVsToPropose[ii.index])
                    #print(f"The reward of {self.index} is:{self.rewardInTimestepGD} and y is {self.y}")
                    self.averageRewardGD = self.averageRewardGD +((self.rewardInTimestepGD - self.averageRewardGD) / timestep)


                else:
                    counterNotAccepted += 1

            if counterNotAccepted == len(listOfUAVs):       #Case 3: The GD proposed to a UAV but did not get accepted
                self.rewardInTimestepGD = 0
                self.preferenceOfUAVsToPropose['No Proposal'] = self.preferenceOfUAVsToPropose['No Proposal'] - self.alpha_ssp * (self.rewardInTimestepGD - self.averageRewardGD) * (self.probabilityOfUAVsToPropose['No Proposal'])
                for UAV in listOfUAVs:
                    if UAV.index == self.chosenUAVtoPropose:
                        self.preferenceOfUAVsToPropose[UAV.index] = self.preferenceOfUAVsToPropose[UAV.index] + self.alpha_ssp * (self.rewardInTimestepGD - self.averageRewardGD) * (1 - self.probabilityOfUAVsToPropose[UAV.index])

                    else:
                        self.preferenceOfUAVsToPropose[UAV.index] = self.preferenceOfUAVsToPropose[UAV.index] - self.alpha_ssp * (self.rewardInTimestepGD - self.averageRewardGD) * (self.probabilityOfUAVsToPropose[UAV.index])

                self.averageRewardGD = self.averageRewardGD + ((self.rewardInTimestepGD - self.averageRewardGD) / timestep)
        self.utilityPerTimestepGD.update({timestep : self.rewardInTimestepGD})
        #print(f"Updated UAV-Preferences of {self.index} are: {self.preferenceOfUAVsToPropose} ")
        return self.rewardInTimestepGD

    def unmatchUAVfromGD(self):
        self.chosenUAVtoPropose = None

        self.distance = 0
        self.channelGain = 0

        self.timeLocalGD = 0
        self.timeCompGD = 0
        self.timeCommGD = 0

        self.rewardInTimestepGD = 0

        self.y = 0






