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
#include "graph.h"
#include "foldermanagement.h"

#include "includespdlog.h"
#include "configreader.h"

struct image_info
{
	std::string image_filename;
	std::string gt_filename;
};

class DataMemory : public QObject
{
	Q_OBJECT

	public:
		explicit DataMemory();
		~DataMemory();
		void configure(QJsonObject const& a_config);
		bool preprocess(QJsonArray m_dataGraph);
		bool loadNamesOfFile();

	public: // get/set
		std::vector<struct image_info> get_imageInfoTrain() { return m_imageInfoTrain; }
		std::vector<struct image_info> get_imageInfoTest() { return m_imageInfoTest; }
		size_t getSizeCleanTrain() { return m_cleanTrain.size(); }
		size_t getSizeGtTrain() { return m_gtTrain.size(); }

		size_t getSizeCleanTest() { return m_cleanTest.size(); }
		size_t getSizeGtTest() { return m_gtTest.size(); }
		
		bool getLoad() { return m_loaded; };
		cv::Mat gt(int i) { return m_gt[i]; }
		cv::Mat clean(int i) { return m_clean[i]; }


	private:
		void loadDataFromStream(cv::VideoCapture videoFromFile, std::vector<cv::Mat>& m_cleanData, bool resize);
		void loadConfig(QJsonObject const& a_config);
		void loadDataFromStreamWindows(std::vector<cv::Mat> &data, bool resize);
		void loadDataLinux(QJsonObject a_config);
		void createSplit();
		void loadGraph(QJsonArray m_dataGraph);
		void clearDataForNextIteration();

	signals:
		void memoryLoaded();

	private:
		std::vector<cv::Mat> m_clean;
		std::vector<cv::Mat> m_gt;
		std::vector<cv::Mat> m_cleanTrain;
		std::vector<cv::Mat> m_cleanTest;
		std::vector<cv::Mat> m_gtTrain;
		std::vector<cv::Mat> m_gtTest;

	private:
		QString m_folderInputPath{};
		QString m_configPath{};
		QString m_cleanPath{};
		QString m_gtPath{};
		QString m_cleanTrainPath{};
		QString m_gtTrainPath{};
		QString m_cleanTestPath{};
		QString m_gtTestPath{};
		QString m_inputType{};
		QString m_outputType{};
		QString m_split{};

	private:
		QJsonArray m_graph;
		std::vector<Processing*> m_block;
		std::vector<std::vector<_data>> m_data;
		std::vector<cv::Mat> m_outputData;
		Graph<Processing, _data> m_graph_processing;
		LoadData m_loadData;
		std::vector<struct image_info>  m_imageInfoTrain;
		std::vector<struct image_info>  m_imageInfoTest;
		
	private:
		int m_width{};
		int m_height{};
		bool m_loaded{};
		bool m_resize{};
		bool m_savePreprocessingDataset{};
		int m_initFrames{};
		int m_zeroPadding{};

		int m_startTrain{};
		int m_stopTrain{};
		int m_startTest{};
		int m_stopTest{};

};

#endif // DATAMEMORY_H
