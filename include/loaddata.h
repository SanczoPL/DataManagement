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
	#ifdef __linux__
		void loadDataFromStream(cv::VideoCapture videoFromFile, std::vector<cv::Mat>& m_cleanData, bool resize);
		void loadDataLinux(std::vector<cv::Mat> &data, std::vector<cv::Mat> &gt);
	#endif // _UNIX

	#ifdef _WIN32
		void loadDataFromStreamWindows(QString path, std::vector<cv::Mat> &data, bool resize);
		void loadDataWindows(std::vector<cv::Mat> &data, std::vector<cv::Mat> &gt);
	#endif // _WIN32

private:
	#ifdef __linux__
	cv::VideoCapture m_videoFromFile;
	cv::VideoCapture m_videoFromFileGT;
	#endif // _UNIX

private:
	QString m_data;
	QString m_gt;
	QString m_inputType;
	QString m_outputType;
	QString m_split;
	QString m_cleanTrain{};
	QString m_gtTrain{};
	QString m_pathToConfig;
	
	int m_stopFrame{};
	int m_startFrame{};
	int m_startGT{};
	int m_stopGT{};
	
private:
	QJsonObject m_datasetConfig;

};

#endif // LOAD_DATA_H
