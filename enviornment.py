import numpy as np
#import random
import groundDevices as gd
import UAV as uav

class Environment:
    def __init__(self, numberOfUAVs, numberOfGDs, mode):
        self.mode = mode
        self.numberOfUAVs = numberOfUAVs
        self.numberOfGDs = numberOfGDs
        self.listOfGDs = []
        self.listOfUAVs = []
        self.timestep = 1


        self.generateGDs()
        self.generateUAVs()

        self.proposalsFromGDtoUAV = {}
        self.proposalsUAV = {}
        self.noProposalGD= []

        self.utilityOfGDInTimestep = 0
        self.utilityOfUAVInTimestep = 0
        self.utilityOfGDs = []
        self.utilityOfUAVs = []



    def generateGDs(self):
        for ii in range(self.numberOfGDs):
            newGD = gd.groundDevices(index=f'GD{ii}', mode=self.mode)
            self.listOfGDs.append(newGD)

    def generateUAVs(self):
        for ii in range(self.numberOfUAVs):
            newUAV = uav.unmannedAerialVehicle(index = f'UAV{ii}', mode=self.mode)
            self.listOfUAVs.append(newUAV)



    def nextStep(self):
        print(f"Timestep: {self.timestep}")
        if self.timestep == 1:
            for GD in self.listOfGDs:
                GD.initializeGD(self.listOfUAVs)

        for GD in self.listOfGDs:
            self.proposalsFromGDtoUAV.update({GD.index:GD.chooseUAVtoPropose(self.listOfUAVs)})
        #print("The Proposals are: ", self.proposalsFromGDtoUAV)
        self.proposalsUAV = {UAV.index: [] for UAV in self.listOfUAVs}
        self.noProposalGD= []

        # Iterate through the proposals dictionary
        for GDs, porposedUAV in self.proposalsFromGDtoUAV.items():
            if porposedUAV == "No Proposal":
                self.noProposalGD.append(GDs)
            else:
                self.proposalsUAV[porposedUAV].append(GDs)

        #print(f"GDs with no proposals: {self.noProposalGD}")
        #print(self.proposalsUAV)

        for UAV in self.listOfUAVs:
            UAV.chooseAcceptedGDs(self.listOfGDs, self.proposalsUAV[UAV.index])

        for GD in self.listOfGDs:
            self.utilityOfGDInTimestep += GD.updateUtilityOfGD(self.listOfUAVs, self.timestep)
        self.utilityOfGDs.append(self.utilityOfGDInTimestep)

        for UAV in self.listOfUAVs:
            self.utilityOfUAVInTimestep += UAV.updateUtilityOfUAVs(self.listOfGDs, self.timestep)
        self.utilityOfUAVs.append(self.utilityOfUAVInTimestep)



        #Unmatch

        for GD in self.listOfGDs:
            GD.unmatchUAVfromGD()
        for UAV in self.listOfUAVs:
            UAV.unmatchGDfromUAV()
        self.proposalsFromGDtoUAV = {}
        self.proposalsUAV = {}
        self.noProposalGD = []
        self.utilityOfGDInTimestep = 0
        self.utilityOfUAVInTimestep = 0

        self.timestep +=1


