import numpy as np


def blockAverage(data: np.ndarray):
    numberOfBlocks = 5
    if isinstance(data, list):
        data = np.array(data)
    assert data.shape[0] > numberOfBlocks

    # create an array that is of size (N/k, k), padded with nans if not divisible
    blockSize = int(np.ceil(data.shape[0] / numberOfBlocks))
    paddedArray = np.full(blockSize * numberOfBlocks, np.nan)
    paddedArray[: data.shape[0]] = data
    paddedArray = paddedArray.reshape(blockSize, numberOfBlocks)

    # accumulate results
    blockAverage = np.nansum(paddedArray, axis=0) / (blockSize - np.sum(np.isnan(paddedArray), axis=0))
    totalAverage = np.mean(blockAverage)
    totalVariance = np.var(blockAverage)
    confidence95 = 2.776 * np.sqrt(totalVariance / numberOfBlocks)
    return totalAverage, confidence95
