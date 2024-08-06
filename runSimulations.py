import numpy as np
import matplotlib.pyplot as plt
import enviornment as env



def run_simulation(monteCarloRunID, T):
    print('Monte Carlo ID:', monteCarloRunID)

    numberOfGDs = 5
    numberOfUAVs = 2
    xPositionGD = []
    yPositionGD = []
    xPositionUAV = []
    yPositionUAV = []
    cpuFrequencyMaxUAV = []
    for ii in range(numberOfGDs):
        xPositionGD.append(int(np.random.uniform(0, 50)))
        yPositionGD.append(int(np.random.uniform(0, 50)))
    for ii in range(numberOfUAVs):
        xPositionUAV.append(int(np.random.uniform(0, 50)))
        yPositionUAV.append(int(np.random.uniform(0, 50)))
        cpuFrequencyMaxUAV.append(int(np.random.uniform(1, 20) * 10 ** 9))

    enviornmentOneSidedLearning = env.Environment(numberOfUAVs, numberOfGDs, xPositionGD, yPositionGD, xPositionUAV, yPositionUAV, cpuFrequencyMaxUAV, mode='oneSidedLearning')
    enviornmentRandom = env.Environment(numberOfUAVs, numberOfGDs, xPositionGD, yPositionGD, xPositionUAV, yPositionUAV, cpuFrequencyMaxUAV, mode='RandomMatching')
    enviornmentGreedy = env.Environment(numberOfUAVs, numberOfGDs, xPositionGD, yPositionGD, xPositionUAV, yPositionUAV, cpuFrequencyMaxUAV, mode='GreedyMatching')
    enviornmentOptimalBenchmark = env.Environment(numberOfUAVs, numberOfGDs, xPositionGD, yPositionGD, xPositionUAV, yPositionUAV, cpuFrequencyMaxUAV, mode='OptimalBenchmark')

    for t in range(T):
        enviornmentOneSidedLearning.nextStep()
        enviornmentRandom.nextStep()
        enviornmentGreedy.nextStep()
        enviornmentOptimalBenchmark.resourceAllocationMatching()

    performanceOfGDsOSL = enviornmentOneSidedLearning.utilityOfGDs
    performanceOfGDsRandom = enviornmentRandom.utilityOfGDs
    performanceOfGDsGreedy = enviornmentGreedy.utilityOfGDs
    performanceOfGDsOptimalBenchmark = enviornmentOptimalBenchmark.utilityOfGDs

    performanceOfUAVsOSL = enviornmentOneSidedLearning.utilityOfUAVs
    performanceOfUAVsRandom = enviornmentRandom.utilityOfUAVs
    performanceOfUAVsGreedy = enviornmentGreedy.utilityOfUAVs
    performanceOfUAVsOptimalBenchmark = enviornmentOptimalBenchmark.utilityOfUAVs

    print(f"Utility of GDs in OSL: {performanceOfGDsOSL}")
    print(f"Utility of GDs in Random: {performanceOfGDsRandom}")
    print(f"Utility of GDs in Greedy: {performanceOfGDsGreedy}")
    print(f"Utility of GDs in OptimalBenchmark: {performanceOfGDsOptimalBenchmark}")
    print('It works so far')

    return performanceOfGDsOSL, performanceOfGDsRandom, performanceOfGDsGreedy, performanceOfGDsOptimalBenchmark, performanceOfUAVsOSL, performanceOfUAVsRandom, performanceOfUAVsGreedy, performanceOfUAVsOptimalBenchmark

def averageResults(performancePerRunDict, meanPerformanceDict):
    for key in performancePerRunDict:
        if performancePerRunDict[key]:
            meanPerformanceDict[key] = np.mean(performancePerRunDict[key])
        else:
            meanPerformanceDict[key] = 0
    return meanPerformanceDict

if __name__ == "__main__":
    T = 200
    monteCarloRuns = 1
    timesteps = list(range(T))

    performanceOfGDsInOSLperRun = {t: [] for t in range(1, T + 1)}
    performanceOfGDsRandomPerRun = {t: [] for t in range(1, T + 1)}
    performanceOfGDsGreedyPerRun = {t: [] for t in range(1, T + 1)}
    performanceOfGDsOptimalBenchmarkPerRun = {t: [] for t in range(1, T + 1)}

    meanPerformanceGDsOSL = {t: 0 for t in range(1, T + 1)}
    meanPerformanceGDsRandom = {t: 0 for t in range(1, T + 1)}
    meanPerformanceGDsGreedy = {t: 0 for t in range(1, T + 1)}
    meanPerformanceGDsOptimalBenchmark = {t: 0 for t in range(1, T + 1)}

    performanceOfUAVsInOSLperRun = {t: [] for t in range(1, T + 1)}
    performanceOfUAVsRandomPerRun = {t: [] for t in range(1, T + 1)}
    performanceOfUAVsGreedyPerRun = {t: [] for t in range(1, T + 1)}
    performanceOfUAVsOptimalBenchmarkPerRun = {t: [] for t in range(1, T + 1)}

    meanPerformanceUAVsOSL = {t: 0 for t in range(1, T + 1)}
    meanPerformanceUAVsRandom = {t: 0 for t in range(1, T + 1)}
    meanPerformanceUAVsGreedy = {t: 0 for t in range(1, T + 1)}
    meanPerformanceUAVsOptimalBenchmark = {t: 0 for t in range(1, T + 1)}

    for m in range(monteCarloRuns):
        gd_osl, gd_Random, gd_Greedy, gd_OptimalBenchmark, uav_osl, uav_Random, uav_Greedy, uav_OptimalBenchmark = run_simulation(m, T)
        for ii in range(len(gd_osl)):
            performanceOfGDsInOSLperRun[ii + 1].append(gd_osl[ii])
            performanceOfGDsRandomPerRun[ii + 1].append(gd_Random[ii])
            performanceOfGDsGreedyPerRun[ii + 1].append(gd_Greedy[ii])
            performanceOfGDsOptimalBenchmarkPerRun[ii + 1].append(gd_OptimalBenchmark[ii])
            performanceOfUAVsInOSLperRun[ii + 1].append(uav_osl[ii])
            performanceOfUAVsRandomPerRun[ii + 1].append(uav_Random[ii])
            performanceOfUAVsGreedyPerRun[ii + 1].append(uav_Greedy[ii])
            performanceOfUAVsOptimalBenchmarkPerRun[ii + 1].append(uav_OptimalBenchmark[ii])

    meanPerformanceGDsOSL = averageResults(performanceOfGDsInOSLperRun, meanPerformanceGDsOSL)
    meanPerformanceGDsRandom = averageResults(performanceOfGDsRandomPerRun, meanPerformanceGDsRandom)
    meanPerformanceGDsGreedy = averageResults( performanceOfGDsGreedyPerRun, meanPerformanceGDsGreedy)
    meanPerformanceGDsOptimalBenchmark = averageResults( performanceOfGDsOptimalBenchmarkPerRun, meanPerformanceGDsOptimalBenchmark)
    meanPerformanceUAVsOSL = averageResults(performanceOfUAVsInOSLperRun, meanPerformanceUAVsOSL)
    meanPerformanceUAVsRandom = averageResults(performanceOfUAVsRandomPerRun, meanPerformanceUAVsRandom)
    meanPerformanceUAVsGreedy = averageResults(performanceOfUAVsGreedyPerRun, meanPerformanceUAVsGreedy)
    meanPerformanceUAVsOptimalBenchmark = averageResults(performanceOfUAVsOptimalBenchmarkPerRun, meanPerformanceUAVsOptimalBenchmark)



    plt.figure()
    plt.plot(meanPerformanceGDsOSL.keys(), meanPerformanceGDsOSL.values(), linestyle='-', color='r', label='OSL')
    plt.plot(meanPerformanceGDsRandom.keys(), meanPerformanceGDsRandom.values(), linestyle='-', color='b', label='Random')
    plt.plot(meanPerformanceGDsGreedy.keys(), meanPerformanceGDsGreedy.values(), linestyle='-', color='g', label='Greedy')
    plt.plot(meanPerformanceGDsOptimalBenchmark.keys(), meanPerformanceGDsOptimalBenchmark.values(), linestyle='-', color='m', label='OptimalBenchmark')

    plt.title('Utility of GDs over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Utility')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(meanPerformanceUAVsOSL.keys(), meanPerformanceUAVsOSL.values(), linestyle='-', color='r', label='OSL')
    plt.plot(meanPerformanceUAVsRandom.keys(), meanPerformanceUAVsRandom.values(), linestyle='-', color='b', label='Random')
    plt.plot(meanPerformanceUAVsGreedy.keys(), meanPerformanceUAVsGreedy.values(), linestyle='-', color='g', label='Greedy')
    plt.plot(meanPerformanceUAVsOptimalBenchmark.keys(), meanPerformanceUAVsOptimalBenchmark.values(), linestyle='-', color='m', label='OptimalBenchmark')

    plt.title('Utility of UAVs over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Utility')
    plt.grid(True)
    plt.legend()
    plt.show()
