#ifndef DATAMEMORY_H
#define DATAMEMORY_H

#include <QObject>
#include <QJsonObject>
#include <QJsonArray>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include "processing.h"
#include "loaddata.h"

#include "includespdlog.h"
#include "configreader.h"


class DataMemory : public QObject {
  Q_OBJECT

public:
	explicit DataMemory();
	~DataMemory();
	void configure(QJsonObject const& a_config);
	bool preprocess(QJsonArray m_dataGraph);
	bool getLoad() { return m_loaded; };

	cv::Mat gt(qint32 i) { return m_gtData[i]; }
	cv::Mat input(qint32 i) { return m_inputData[i]; }

	qint32 getSize() { return m_inputData.size(); }
	qint32 getSizeGT() { return m_gtData.size(); }

	void loadGraph(QJsonArray m_dataGraph);
	void clearDataForNextIteration();
	bool checkIfLoadInputs(const int i, const QJsonArray _prevActive, std::vector<_data> & dataVec, std::vector<cv::Mat> &input);
	bool checkIfReturnData(const QJsonArray _nextActive);
	void loadInputs(const QJsonArray _prevActive, std::vector<_data> & dataVec);
	void returnData(int i, std::vector<cv::Mat> & m_outputData);

private:
	void loadDataFromStream(cv::VideoCapture videoFromFile, std::vector<cv::Mat>& m_cleanData, bool resize);
	void loadConfig(QJsonObject const& a_config);
	QVector<QString> scanAllImages(QString path);
	void loadDataFromStreamWindows(std::vector<cv::Mat> &data, bool resize);
	void loadDataLinux(QJsonObject a_config);
	void createSplit();

signals:
  void memoryLoaded();

private:
	std::vector<cv::Mat> m_cleanData;
	std::vector<cv::Mat> m_gtCleanData;
	std::vector<cv::Mat> m_inputData;
	std::vector<cv::Mat> m_gtData;

private:
	QString m_folderInput;
	QString m_roi;
	qint32 m_stopFrame{};
	qint32 m_startFrame{};
	qint32 m_startGT{};
	qint32 m_stopGT{};
	qint32 m_width;
	qint32 m_height;
	QString m_clean;
	QString m_gt;

private:
	QJsonArray m_graph;
	std::vector<Processing*> m_block;
	std::vector<std::vector<_data>> m_data;
	std::vector<cv::Mat> m_outputData;

private:
	QString m_inputType;
	QString m_outputType;
	QString m_split;

	Graph<Processing, _data> m_graph_processing;
	LoadData m_loadData;

	bool m_loaded{};
	bool m_resize{};
	bool m_savePreprocessingDataset{};
	QString m_pathToConfig;
	QJsonObject m_datasetConfig;
	QString m_cleanTrain{};
	QString m_gtTrain{};

};

#endif // DATAMEMORY_H
