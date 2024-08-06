import numpy as np
import groundDevices as gd
import UAV as uav
import copy

class Environment:
    def __init__(self, numberOfUAVs, numberOfGDs, xPositionGD, yPositionGD, xPositionUAV, yPositionUAV, cpuFrequencyMaxUAV, mode, debug = True):
        self.mode = mode
        self.numberOfUAVs = numberOfUAVs
        self.numberOfGDs = numberOfGDs
        self.xPositionGD = xPositionGD
        self.yPositionGD = yPositionGD
        self.xPositionUAV = xPositionUAV
        self.yPositionUAV = yPositionUAV
        self.cpuFrequencyMaxUAV = cpuFrequencyMaxUAV
        self.listOfGDs = []
        self.listOfUAVs = []
        self.timestep = 1

        self.generateGDs()
        self.generateUAVs()

        self.proposalsFromGDtoUAV = {}
        self.proposalsUAV = {}
        self.noProposalGD = []

        self.utilityOfGDInTimestep = 0
        self.utilityOfUAVInTimestep = 0
        self.utilityOfGDs = []
        self.utilityOfUAVs = []
        self.debug = debug

    def generateGDs(self):
        for ii in range(self.numberOfGDs):
            newGD = gd.groundDevices(index=f'GD{ii}', xPositionGD=self.xPositionGD[ii], yPositionGD=self.yPositionGD[ii], mode=self.mode)
            self.listOfGDs.append(newGD)

    def generateUAVs(self):
        for ii in range(self.numberOfUAVs):
            newUAV = uav.unmannedAerialVehicle(index=f'UAV{ii}', xPositionUAV=self.xPositionUAV[ii], yPositionUAV=self.yPositionUAV[ii], cpuFrequencyMaxUAV=self.cpuFrequencyMaxUAV[ii], mode=self.mode)
            self.listOfUAVs.append(newUAV)

    def nextStep(self):
        print(f"Timestep: {self.timestep}")
        if self.timestep == 1:
            for GD in self.listOfGDs:
                GD.initializeGD(self.listOfUAVs)

        for GD in self.listOfGDs:
            self.proposalsFromGDtoUAV.update({GD.index: GD.chooseUAVtoPropose(self.listOfUAVs)})

        self.proposalsUAV = {UAV.index: [] for UAV in self.listOfUAVs}
        self.noProposalGD = []

        for GDs, proposedUAV in self.proposalsFromGDtoUAV.items():
            if proposedUAV == "No Proposal":
                self.noProposalGD.append(GDs)
            else:
                self.proposalsUAV[proposedUAV].append(GDs)

        for UAV in self.listOfUAVs:
            UAV.chooseAcceptedGDs(self.listOfGDs, self.proposalsUAV[UAV.index])

        for GD in self.listOfGDs:
            self.utilityOfGDInTimestep += GD.updateUtilityOfGD(self.listOfUAVs, self.timestep)
        self.utilityOfGDs.append(self.utilityOfGDInTimestep)

        for UAV in self.listOfUAVs:
            self.utilityOfUAVInTimestep += UAV.updateUtilityOfUAVs(self.listOfGDs, self.timestep)
        self.utilityOfUAVs.append(self.utilityOfUAVInTimestep)

        self.clearMatching()

    def clearMatching(self):
        for GD in self.listOfGDs:
            GD.unmatchUAVfromGD(self.listOfUAVs)
            GD.initializeGDRAM
        for UAV in self.listOfUAVs:
            UAV.unmatchGDfromUAV()
        self.proposalsFromGDtoUAV = {}
        self.proposalsUAV = {}
        self.noProposalGD = []
        self.utilityOfGDInTimestep = 0
        self.utilityOfUAVInTimestep = 0
        self.timestep += 1

    def resourceAllocationMatching(self):
        if self.debug:
            print(f"Timestep: {self.timestep}")
        for GD in self.listOfGDs:
            GD.initializeGD(self.listOfUAVs)
            GD.initializeGDRAM(self.listOfUAVs)
        self.clearMatching()

        unstable = True

        while unstable:
            # Set the matching to stable, this will be changed if one GD is unstable
            unstable = False
            for GD in self.listOfGDs:
                # Skip GD if it is already matched or has no entries in its UAV-preference-list
                if GD.matchedUAVRAM or not GD.possibleUAVsRAM:
                    continue

                # Step 1: Create a Preference List of all UAVs
                GD.createPreferenceListRAM(self.listOfUAVs, GD.possibleUAVsRAM)
                GD.chooseUAVtoProposeRAM = max(GD.preferenceListUAVsRAM, key=GD.preferenceListUAVsRAM.get)
                GD.provideInformationToUAV(GD.chooseUAVtoProposeRAM, self.listOfUAVs)
                if self.debug:
                    print(f"GD {GD.index} proposing to UAV {GD.chooseUAVtoProposeRAM} with preference {GD.preferenceListUAVsRAM[GD.chooseUAVtoProposeRAM]}")

                if GD.timeLocalGD > GD.preferenceListUAVsRAM[GD.chooseUAVtoProposeRAM] > 0:
                    unstable = True
                    # Use the index directly to append to proposingGDs
                    uav_index = next(index for index, UAV in enumerate(self.listOfUAVs) if UAV.index == GD.chooseUAVtoProposeRAM)
                    self.listOfUAVs[uav_index].proposingGDs.append(GD)
                    if GD.chooseUAVtoProposeRAM in GD.possibleUAVsRAM:
                        GD.possibleUAVsRAM.remove(GD.chooseUAVtoProposeRAM)
                    else:
                        print(f"Error: {GD.chooseUAVtoProposeRAM} not found in GD.possibleUAVsRAM for GD {GD.index}")

            for UAV in self.listOfUAVs:
                if not UAV.proposingGDs:
                    continue
                if self.debug:
                    print(f"UAV {UAV.index} has the proposing GDs: {[gd.index for gd in UAV.proposingGDs]}")

                I = copy.deepcopy(UAV.listOfAcceptedGDs)
                sigma = copy.deepcopy(UAV.proposingGDs)
                marginalContributions = {}

                for GD in sigma:
                    marginalContributions[GD.index] = UAV.marginalUtility([GD])
                while marginalContributions:
                    max_key = max(marginalContributions, key=marginalContributions.get)
                    A = []
                    for k in I:
                        temp_I = list(set(I) - set(A)) + [max_key]
                        if UAV.marginalUtility(
                                [gd for gd in self.listOfGDs if gd.index in temp_I]) > UAV.marginalUtility(
                                [gd for gd in self.listOfGDs if gd.index in I]):
                            continue
                        X = list(set(I) - set(A)) + [max_key]
                        X.remove(k)
                        if UAV.marginalUtility([gd for gd in self.listOfGDs if gd.index in X]) > UAV.marginalUtility(
                                [gd for gd in self.listOfGDs if gd.index in I]):
                            A.append(k)
                    if UAV.marginalUtility([gd for gd in self.listOfGDs if
                                            gd.index in (list(set(I) - set(A)) + [max_key])]) > UAV.marginalUtility(
                            [gd for gd in self.listOfGDs if gd.index in I]):
                        I = list(set(I) - set(A)) + [max_key]
                        marginalContributions.pop(max_key)
                    else:
                        break

                for GD in list(set(UAV.listOfAcceptedGDs) - set(I)):
                    self.listOfGDs[GD].unmatchUAVfromGD(self.listOfUAVs)

                UAV.listOfAcceptedGDs = I
                if self.debug:
                    print(f"UAV {UAV.index} has the accepted GDs: {UAV.listOfAcceptedGDs}")
                for GD in I:
                    gd_index = next(index for index, gd in enumerate(self.listOfGDs) if gd.index == GD)
                    self.listOfGDs[gd_index].matchedUAVRAM = UAV.index

                UAV.proposingGDs = []

            for GD in self.listOfGDs:
                self.utilityOfGDInTimestep += GD.updateUtilityOfGD(self.listOfUAVs, self.timestep)
            self.utilityOfGDs.append(self.utilityOfGDInTimestep)
            if self.debug:
                print(f"Utility of GDs at timestep {self.timestep}: {self.utilityOfGDInTimestep}")

            for UAV in self.listOfUAVs:
                self.utilityOfUAVInTimestep += UAV.updateUtilityOfUAVs(self.listOfGDs, self.timestep)
            self.utilityOfUAVs.append(self.utilityOfUAVInTimestep)
            if self.debug:
                print(f"Utility of UAVs at timestep {self.timestep}: {self.utilityOfUAVInTimestep}")

            return

