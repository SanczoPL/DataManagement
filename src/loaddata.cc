#include "loaddata.h"

constexpr auto CLEAN{ "Clean" };
constexpr auto GT{ "Gt" };
constexpr auto INPUT_TYPE{ "InputType" };
constexpr auto OUTPUT_TYPE{ "OutputType" };
constexpr auto INPUT_PREFIX{ "InputPrefix" };
constexpr auto DATASET_UNIX{ "DatasetLinux" };
constexpr auto DATASET_WIN32{ "DatasetWin32" };
constexpr auto START_GT{"StartGT"};
constexpr auto STOP_GT{"StopGT"};
constexpr auto PATH_TO_DATASET{ "PathToDataset" };
constexpr auto CONFIG_NAME{ "ConfigName" };
constexpr auto CLEAN_TRAIN{ "Clean_train" };
constexpr auto GT_TRAIN{ "Gt_train" };
constexpr auto SAVE_PREPROCESSING_DATASET{ "SavePreprocessingDataset" };

//#define DEBUG
//#define DEBUG_PREPROCESS
//#define DEBUG_OPENCV


LoadData::LoadData()
{
	#ifdef DEBUG
	Logger->debug("LoadData::LoadData()");
	#endif
}

LoadData::~LoadData(){}

void LoadData::configure(QJsonObject const& a_config)
{
	#ifdef _WIN32
	m_split = "\\";
	#endif // _WIN32
	#ifdef __linux__
	m_split = "/";
	#endif // _UNIX

	#ifdef _WIN32
	QJsonObject jDataset{ a_config[DATASET_WIN32].toObject() };
	#endif // _WIN32
	#ifdef __linux__ 
	QJsonObject jDataset{ a_config[DATASET_UNIX].toObject() };
	#endif // _UNIX
	
	QString configName = jDataset[CONFIG_NAME].toString();
	m_pathToConfig = jDataset[PATH_TO_DATASET].toString();


	QJsonObject datasetConfig{};
	std::shared_ptr<ConfigReader> cR = std::make_shared<ConfigReader>();
	if (!cR->readConfig(m_pathToConfig + configName, datasetConfig))
	{
		Logger->error("DataMemory::configure() File {} not readed", (m_pathToConfig + configName).toStdString());
	}

	m_datasetConfig = datasetConfig;

	m_startGT = datasetConfig[START_GT].toInt();
	m_stopGT = datasetConfig[STOP_GT].toInt();
	m_inputType = datasetConfig[INPUT_TYPE].toString();
	m_outputType = datasetConfig[OUTPUT_TYPE].toString();
	m_cleanTrain = datasetConfig[CLEAN_TRAIN].toString();
	m_gtTrain = datasetConfig[GT_TRAIN].toString();
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
	
	Logger->debug("LoadData::loadData() gt.size data:{}", (gt).size());
	#endif
	Logger->debug("LoadData::loadData() data.size:{} ({}x{})", (data).size(), data[0].cols, data[0].rows);

	return true;
}

#ifdef __linux__
void LoadData::loadDataFromStream(cv::VideoCapture videoFromFile, std::vector<cv::Mat> &data, bool resize)
{
	#ifdef DEBUG
	Logger->debug("LoadData::loadDataFromStream()");
	#endif
	int iter{ 0 };
	while (videoFromFile.isOpened())
	{
		cv::Mat inputMat;
		videoFromFile >> inputMat;
		if (resize && inputMat.channels() > 1)
		{
			cv::cvtColor(inputMat, inputMat, 6);
		}
		iter++;
		if (inputMat.cols == 0 || inputMat.rows == 0 || iter >= m_stopGT)
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
	m_data = m_pathToConfig + m_datasetConfig[CLEAN].toString() + m_split + m_datasetConfig[INPUT_PREFIX].toString();
	m_gt = m_pathToConfig + m_datasetConfig[GT].toString() + m_split + m_datasetConfig[INPUT_PREFIX].toString();
	//#ifdef DEBUG
	Logger->debug("m_data:{}", m_data.toStdString());
	Logger->debug("m_gt:{}", m_gt.toStdString());

	//#endif
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
	
	loadDataFromStream(m_videoFromFile, data, true);
	loadDataFromStream(m_videoFromFileGT, gt, true);
}
#endif

#ifdef _WIN32

void LoadData::loadDataFromStreamWindows(QString path, std::vector<cv::Mat> &data, bool resize)
{
	QVector<QString> m_imgList = scanAllImages(path);
	std::sort(m_imgList.begin(), m_imgList.end());
	Logger->trace("m_imgList:{}", m_imgList.size());
	if (m_imgList.size() > 0)
	{
		for (qint32 iteration = 0; iteration < m_imgList.size(); iteration++)
		{
			if (iteration % 100 == 0)
			{
				Logger->info("iteration:{}", iteration);
			}
			QString name = path +  m_split + m_imgList[iteration] + m_inputType;

			cv::Mat inputMat = cv::imread((name).toStdString(), cv::IMREAD_GRAYSCALE);
			data.push_back(inputMat);

			if(iteration == m_stopGT)
			{
				Logger->info("LoadData::loadDataFromStreamWindows() stop loading on:{}", iteration);
				break;
			}
		}
	}
}

void LoadData::loadDataWindows(std::vector<cv::Mat> &data, std::vector<cv::Mat> &gt)
{
	m_data = m_pathToConfig + m_datasetConfig[CLEAN].toString();
	m_gt = m_pathToConfig + m_datasetConfig[GT].toString();

	#ifdef DEBUG
		Logger->debug("LoadData::loadDataWindows() m_data:{}", m_data.toStdString());
		Logger->debug("LoadData::loadDataWindows() m_gt:{}", m_gt.toStdString());
	#endif

	loadDataFromStreamWindows(m_data, data, true);
	loadDataFromStreamWindows(m_gt, gt, true);
}

QVector<QString> LoadData::scanAllImages(QString path)
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
#endif
