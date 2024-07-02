import numpy as np
import matplotlib.pyplot as plt
import enviornment as env

def run_simulation(monteCarloRunID, T):
    print('Monte Carlo ID:', monteCarloRunID)

    numberOfGDs = 15
    numberOfUAVs = 4


    enviornmentOneSidedLearning = env.Environment(numberOfUAVs, numberOfGDs, mode='oneSidedLearning')
    enviornmentUAVsOnly = env.Environment(numberOfUAVs, numberOfGDs, mode='UAVsOnlyMatching')
    enviornmentChannelGainBased = env.Environment(numberOfUAVs, numberOfGDs, mode='ChannelGainBasedMatching')
    for t in range(T):
        enviornmentOneSidedLearning.nextStep()
        enviornmentUAVsOnly.nextStep()
        enviornmentChannelGainBased.nextStep()


    performanceOfGDsOSL = enviornmentOneSidedLearning.utilityOfGDs
    performanceOfGDsUAVsOnly = enviornmentUAVsOnly.utilityOfGDs
    performanceOfGDsChannelGainBased = enviornmentChannelGainBased.utilityOfGDs
    performanceOfUAVsOSL = enviornmentOneSidedLearning.utilityOfUAVs
    performanceOfUAVsUAVsOnly = enviornmentUAVsOnly.utilityOfUAVs
    performanceOfUAVsChannelGainBased = enviornmentChannelGainBased.utilityOfUAVs


    print(f"Utility of GDs in OSL: {enviornmentOneSidedLearning.utilityOfGDs}")
    print(f"Utility of GDs in UAVsOnly: {enviornmentUAVsOnly.utilityOfGDs}")
    print(f"Utility of GDs in ChannelGainBased: {enviornmentChannelGainBased.utilityOfGDs}")
    print('it works so far')

    return performanceOfGDsOSL, performanceOfGDsUAVsOnly, performanceOfGDsChannelGainBased, performanceOfUAVsOSL, performanceOfUAVsUAVsOnly, performanceOfUAVsChannelGainBased


if __name__ == "__main__":
    T = 200
    monteCarloRuns = 1
    timesteps = list(range(T))

    performanceOfGDsInOSLperRun = {t: [] for t in range(1, T + 1)}
    performanceOfGDsUAVsOnlyPerRun = {t: [] for t in range(1, T + 1)}
    performanceOfGDsChannelGainBasedPerRun = {t: [] for t in range(1, T + 1)}

    meanPerformanceGDsOSL = {t: 0 for t in range(1, T + 1)}
    meanPerformanceGDsUAVsOnly = {t: 0 for t in range(1, T + 1)}
    meanPerformanceGDsChannelGainBased = {t: 0 for t in range(1, T + 1)}

    performanceOfUAVsInOSLperRun = {t: [] for t in range(1, T + 1)}
    performanceOfUAVsUAVsOnlyPerRun = {t: [] for t in range(1, T + 1)}
    performanceOfUAVsChannelGainBasedPerRun = {t: [] for t in range(1, T + 1)}

    meanPerformanceUAVsOSL = {t: 0 for t in range(1, T + 1)}
    meanPerformanceUAVsUAVsOnly = {t: 0 for t in range(1, T + 1)}
    meanPerformanceUAVsChannelGainBased = {t: 0 for t in range(1, T + 1)}


    # Lists to store the results
    results_list = []
    for m in range(monteCarloRuns):
        gd_osl, gd_UAVsOnly, gd_ChannelGainBased, uav_osl, uav_UAVsOnly, uav_ChannelGainBased = run_simulation(m,T)
        for ii in range(len(gd_osl)):
            performanceOfGDsInOSLperRun[ii + 1].append(gd_osl[ii])
            performanceOfGDsUAVsOnlyPerRun[ii + 1].append(gd_UAVsOnly[ii])
            performanceOfGDsChannelGainBasedPerRun[ii + 1].append(gd_ChannelGainBased[ii])
            performanceOfUAVsInOSLperRun[ii + 1].append(uav_osl[ii])
            performanceOfUAVsUAVsOnlyPerRun[ii + 1].append(uav_UAVsOnly[ii])
            performanceOfUAVsChannelGainBasedPerRun[ii + 1].append(uav_ChannelGainBased[ii])



    for key in performanceOfGDsInOSLperRun:
        if performanceOfGDsInOSLperRun[key]:
            meanPerformanceGDsOSL[key] = np.mean(performanceOfGDsInOSLperRun[key])
        else:
            meanPerformanceGDsOSL[key] = 0

    for key in performanceOfGDsUAVsOnlyPerRun:
        if performanceOfGDsUAVsOnlyPerRun[key]:
            meanPerformanceGDsUAVsOnly[key] = np.mean(performanceOfGDsUAVsOnlyPerRun[key])
        else:
            meanPerformanceGDsUAVsOnly[key] = 0

    for key in performanceOfGDsChannelGainBasedPerRun:
        if performanceOfGDsChannelGainBasedPerRun[key]:
            meanPerformanceGDsChannelGainBased[key] = np.mean(performanceOfGDsChannelGainBasedPerRun[key])
        else:
            meanPerformanceGDsChannelGainBased[key] = 0

    for key in performanceOfUAVsInOSLperRun:
        if performanceOfUAVsInOSLperRun[key]:
            meanPerformanceUAVsOSL[key] = np.mean(performanceOfUAVsInOSLperRun[key])
        else:
            meanPerformanceUAVsOSL[key] = 0

    for key in performanceOfUAVsUAVsOnlyPerRun:
        if performanceOfUAVsUAVsOnlyPerRun[key]:
            meanPerformanceUAVsUAVsOnly[key] = np.mean(performanceOfUAVsUAVsOnlyPerRun[key])
        else:
            meanPerformanceUAVsUAVsOnly[key] = 0

    for key in performanceOfUAVsChannelGainBasedPerRun:
        if performanceOfUAVsChannelGainBasedPerRun[key]:
            meanPerformanceUAVsChannelGainBased[key] = np.mean(performanceOfUAVsChannelGainBasedPerRun[key])
        else:
            meanPerformanceUAVsChannelGainBased[key] = 0




    plt.figure()
    plt.plot(meanPerformanceGDsOSL.keys(), meanPerformanceGDsOSL.values(), linestyle='-', color='r', label = 'OSL')
    plt.plot(meanPerformanceGDsUAVsOnly.keys(), meanPerformanceGDsUAVsOnly.values(), linestyle='-', color='b', label='UAVsOnly')
    plt.plot(meanPerformanceGDsChannelGainBased.keys(), meanPerformanceGDsChannelGainBased.values(), linestyle='-', color='g', label='ChannelGainBased')

    # Add titles and labels
    plt.title('Utility of GDs over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Utility')

    # Show the plot
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(meanPerformanceUAVsOSL.keys(), meanPerformanceUAVsOSL.values(), linestyle='-', color='r', label='OSL')
    plt.plot(meanPerformanceUAVsUAVsOnly.keys(), meanPerformanceUAVsUAVsOnly.values(), linestyle='-', color='b', label='UAVsOnly')
    plt.plot(meanPerformanceUAVsChannelGainBased.keys(), meanPerformanceUAVsChannelGainBased.values(), linestyle='-', color='g', label='ChannelGainBased')

    # Add titles and labels
    plt.title('Utility of UAVs over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Utility')

    # Show the plot
    plt.grid(True)
    plt.legend()
    plt.show()