#include "loaddata.h"

constexpr auto WIDTH{ "Width" };
constexpr auto HEIGHT{ "Height" };
constexpr auto DIR_CLEAN{ "DirectoryClean" };
constexpr auto DIR_GT{ "DirectoryGt" };
constexpr auto DIR_CLEAN_TRAIN{ "DirectoryCleanTrain" };
constexpr auto DIR_CLEANT_TEST{ "DirectoryCleanTest" };
constexpr auto DIR_GT_TRAIN{ "DirectoryGtTrain" };
constexpr auto DIR_GT_TEST{ "DirectoryGtTest" };
constexpr auto START_TRAIN{"StartTrain"};
constexpr auto STOP_TRAIN{"StopTrain"};
constexpr auto START_TEST{"StartTest"};
constexpr auto STOP_TEST{"StopTest"};
constexpr auto RESIZE{"Resize"};
constexpr auto INPUT_TYPE{ "InputType" };
constexpr auto OUTPUT_TYPE{ "OutputType" };
constexpr auto INPUT_PREFIX{ "InputPrefix" };
constexpr auto INITIALIZATION_FRAMES{ "InitializationFrames" };
constexpr auto ALL_FRAMES{ "AllFrames" };

constexpr auto DATASET_UNIX{ "DatasetLinux" };
constexpr auto DATASET_WIN32{ "DatasetWin32" };
constexpr auto PATH_TO_DATASET{ "PathToDataset" };
constexpr auto CONFIG_NAME{ "ConfigName" };
constexpr auto SAVE_PREPROCESSING_DATASET{ "SavePreprocessingDataset" };

//#define DEBUG
//#define DEBUG_PREPROCESS
//#define DEBUG_OPENCV


LoadData::LoadData()
{
	#ifdef DEBUG
	Logger->debug("LoadData::LoadData()");
	#endif
	LoadData::createSplit();
}

LoadData::~LoadData(){}

void LoadData::createSplit()
{
	#ifdef DEBUG
	Logger->debug("LoadData::createSplit()");
	#endif
	#ifdef _WIN32
	m_split = "\\";
	#endif // _WIN32
	#ifdef __linux__
	m_split = "/";
	#endif // _UNIX
}

void LoadData::configure(QJsonObject const& a_config)
{
	#ifdef DEBUG
	Logger->debug("LoadData::configure()");
	#endif
	#ifdef _WIN32
	QJsonObject jDataset{ a_config[DATASET_WIN32].toObject() };
	#endif // _WIN32
	#ifdef __linux__ 
	QJsonObject jDataset{ a_config[DATASET_UNIX].toObject() };
	#endif // _UNIX
	
	QString configName = jDataset[CONFIG_NAME].toString();
	m_configPath = jDataset[PATH_TO_DATASET].toString();


	QJsonObject datasetConfig{};
	std::shared_ptr<ConfigReader> cR = std::make_shared<ConfigReader>();
	if (!cR->readConfig(m_configPath + configName, datasetConfig))
	{
		Logger->error("DataMemory::configure() File {} not readed", (m_configPath + configName).toStdString());
	}

	m_datasetConfig = datasetConfig;

	m_cleanPath = datasetConfig[DIR_CLEAN].toString();
	m_gtPath = datasetConfig[DIR_GT].toString();

	m_inputType = datasetConfig[INPUT_TYPE].toString();
	m_outputType = datasetConfig[OUTPUT_TYPE].toString();

	m_cleanTrainPath = datasetConfig[DIR_CLEAN_TRAIN].toString();
	m_gtTrainPath = datasetConfig[DIR_GT_TRAIN].toString();

	m_cleanTestPath = datasetConfig[DIR_CLEANT_TEST].toString();
	m_gtTestPath = datasetConfig[DIR_GT_TEST].toString();

	m_startTrain = datasetConfig[START_TRAIN].toInt();
	m_stopTrain = datasetConfig[STOP_TRAIN].toInt();
	m_startTest = datasetConfig[START_TEST].toInt();
	m_stopTest = datasetConfig[STOP_TEST].toInt();

	m_allFrames = datasetConfig[ALL_FRAMES].toInt();

	#ifdef DEBUG
	Logger->debug("LoadData::configure() done");
	#endif
}

bool LoadData::loadData(std::vector<cv::Mat> &data, std::vector<cv::Mat> &gt)
{
	data.clear();
	gt.clear();

	#ifdef __linux__ 
		LoadData::loadDataLinux(data, gt);
	#endif // _UNIX

	#ifdef _WIN32
		LoadData::loadDataWindows(data, gt);
	#endif // _WIN32

	#ifdef DEBUG
	Logger->debug("LoadData::loadData() data.size:{} ({}x{})", (data).size(), data[0].cols, data[0].rows);
	Logger->debug("LoadData::loadData() gt.size data:{}", (gt).size());
	#endif
	return true;
}

#ifdef __linux__
void LoadData::loadDataFromStream(cv::VideoCapture videoFromFile, std::vector<cv::Mat> &data, int framesNumber)
{
	#ifdef DEBUG
	Logger->debug("LoadData::loadDataFromStream()");
	#endif
	int iter{ 0 };
	while (videoFromFile.isOpened())
	{
		cv::Mat inputMat;
		videoFromFile >> inputMat;
		if (inputMat.channels() > 1)
		{
			cv::cvtColor(inputMat, inputMat, 6);
		}
		iter++;
		if (inputMat.cols == 0 || inputMat.rows == 0 || iter > framesNumber)
		{
			#ifdef DEBUG
			Logger->debug("push_back {} images end", iter);
			#endif
			break;
		}
		data.push_back(inputMat);
	}
}

void LoadData::loadDataLinux(std::vector<cv::Mat> &data, std::vector<cv::Mat> &gt)
{
	QString m_data = m_configPath + m_datasetConfig[CLEAN].toString() + m_split + m_datasetConfig[INPUT_PREFIX].toString();
	QString m_gt = m_configPath + m_datasetConfig[GT].toString() + m_split + m_datasetConfig[INPUT_PREFIX].toString();
	#ifdef DEBUG
		Logger->debug("m_data:{}", m_data.toStdString());
		Logger->debug("m_gt:{}", m_gt.toStdString());
	#endif
	int ret = m_videoFromFile.open(m_data.toStdString());
	if (ret < 0)
	{
		Logger->error("input data failed to open:{}", (m_data).toStdString());
	}

	ret = m_videoFromFileGT.open(m_gt.toStdString());
	if (ret < 0)
	{
		Logger->error("input data failed to open:{}", (m_gt).toStdString());
	}
	
	loadDataFromStream(m_videoFromFile, data, m_allFrames);
	loadDataFromStream(m_videoFromFileGT, gt, m_allFrames);
}
#endif

#ifdef _WIN32

void LoadData::loadDataFromStreamWindows(QString path, std::vector<cv::Mat> &data, int framesNumber)
{
	QVector<QString> m_imgList = scanAllImages(path);
	std::sort(m_imgList.begin(), m_imgList.end());
	#ifdef DEBUG
		Logger->debug("m_imgList:{}", m_imgList.size());
	#endif

	if (m_imgList.size() > 0)
	{
		for (qint32 iteration = 0; iteration < m_imgList.size(); iteration++)
		{
			if (iteration % 100 == 0)
			{
				Logger->trace("loadDataFromStreamWindows() loaded frames:{}", iteration);
			}
			QString name = path +  m_split + m_imgList[iteration] + m_inputType;

			cv::Mat inputMat = cv::imread((name).toStdString(), cv::IMREAD_GRAYSCALE);
			data.push_back(inputMat);

			if(iteration > framesNumber)
			{
				Logger->info("LoadData::loadDataFromStreamWindows() stop loading on:{} frame", iteration);
				break;
			}
		}
	}
}

void LoadData::loadDataWindows(std::vector<cv::Mat> &data, std::vector<cv::Mat> &gt)
{
	QString m_data = m_configPath + m_cleanPath;
	QString m_gt = m_configPath + m_gtPath;

	#ifdef DEBUG
		Logger->debug("LoadData::loadDataWindows() m_data:{}", m_data.toStdString());
		Logger->debug("LoadData::loadDataWindows() m_gt:{}", m_gt.toStdString());
	#endif

	loadDataFromStreamWindows(m_data, data, m_allFrames);
	loadDataFromStreamWindows(m_gt, gt, m_allFrames);
}

#endif
