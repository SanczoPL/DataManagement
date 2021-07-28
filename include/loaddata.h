#ifndef LOAD_DATA_H
#define LOAD_DATA_H

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
#include "graph.h"

#include "includespdlog.h"
#include "configreader.h"

#ifdef _WIN32
#endif // _WIN32
#ifdef __linux__
#endif // _UNIX

QVector<QString> static scanAllImages(QString path)
{
    QVector<QString> temp;
    QDir directory(path);
    QStringList images = directory.entryList(QStringList() << "*.jpg" << "*.png" << "*.PNG" << "*.JPG", QDir::Files);

    foreach(QString filename, images)
    {
        QStringList sl = filename.split(".");
        temp.push_back(sl[0]);
    }
    return temp;
}
QVector<QString> static scanAllVideo(QString path)
{
    QVector<QString> temp;
    QDir directory(path);
    QStringList images = directory.entryList(QStringList() << "*.MP4" << "*.mp4", QDir::Files);

    foreach(QString filename, images)
    {
        QStringList sl = filename.split(".");
        temp.push_back(sl[0]);
    }
    return temp;
}

class LoadData : public QObject {
	Q_OBJECT

public:
	LoadData();
	~LoadData();
	bool loadData(std::vector<cv::Mat> &data, std::vector<cv::Mat> &gt);
	void configure(QJsonObject const& a_config);

private:
	void createSplit();
	#ifdef __linux__
		void loadDataFromStream(cv::VideoCapture videoFromFile, std::vector<cv::Mat> &data, int framesNumber);
		void loadDataLinux(std::vector<cv::Mat> &data, std::vector<cv::Mat> &gt);
	#endif // _UNIX

	#ifdef _WIN32
		void loadDataFromStreamWindows(QString path, std::vector<cv::Mat> &data, int framesNumber);
		void loadDataWindows(std::vector<cv::Mat> &data, std::vector<cv::Mat> &gt);
	#endif // _WIN32

private:
	#ifdef __linux__
	cv::VideoCapture m_videoFromFile;
	cv::VideoCapture m_videoFromFileGT;
	#endif // _UNIX
	
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
	int m_startTrain{};
	int m_stopTrain{};
	int m_startTest{};
	int m_stopTest{};
	int m_allFrames{};
	
private:
	QJsonObject m_datasetConfig;

};

#endif // LOAD_DATA_H
