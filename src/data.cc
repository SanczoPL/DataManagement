#include "data.h"

//#define DEBUG
//#define DEBUG_PREPROCESS
//#define DEBUG_OPENCV

constexpr auto GRAPH{ "Graph" };
constexpr auto NAME{ "Name" };
constexpr auto ACTIVE{ "Active" };
constexpr auto COPY_SIGNAL{ "Signal" };
constexpr auto TYPE{ "Type" };
constexpr auto NEXT{ "Next" };
constexpr auto PREV{ "Prev" };
constexpr auto CONFIG{ "Config" };

//Configs:
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
constexpr auto ZERO_PADDING{ "ZeroPadding" };

constexpr auto PATH_TO_DATASET{ "PathToDataset" };
constexpr auto CONFIG_NAME{ "ConfigName" };
constexpr auto DATASET_UNIX{ "DatasetLinux" };
constexpr auto DATASET_WIN32{ "DatasetWin32" };
constexpr auto SAVE_PREPROCESSING_DATASET{ "SavePreprocessingDataset" };


DataMemory::DataMemory()
{
	#ifdef DEBUG
	Logger->debug("DataMemory::DataMemory()");
	#endif
	DataMemory::createSplit();
}

DataMemory::~DataMemory(){}

void DataMemory::createSplit()
{
	#ifdef DEBUG
	Logger->debug("DataMemory::createSplit()");
	#endif
	#ifdef _WIN32
	m_split = "\\";
	#endif // _WIN32
	#ifdef __linux__
	m_split = "/";
	#endif // _UNIX
}

void DataMemory::clearDataForNextIteration()
{
	m_data.clear();
	m_outputData.clear();
}

void DataMemory::loadConfig(QJsonObject const& a_config)
{
	#ifdef DEBUG
	Logger->debug("DataMemory::loadConfig()");
	qDebug() << "DataMemory::loadConfig() config:"<< a_config;
	#endif
	m_savePreprocessingDataset = a_config[SAVE_PREPROCESSING_DATASET].toBool();

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

	m_cleanPath = datasetConfig[DIR_CLEAN].toString();
	m_gtPath = datasetConfig[DIR_GT].toString();

	m_inputType = datasetConfig[INPUT_TYPE].toString();
	m_outputType = datasetConfig[OUTPUT_TYPE].toString();

	m_cleanTrainPath = datasetConfig[DIR_CLEAN_TRAIN].toString();
	m_gtTrainPath = datasetConfig[DIR_GT_TRAIN].toString();

	m_cleanTestPath = datasetConfig[DIR_CLEANT_TEST].toString();
	m_gtTestPath = datasetConfig[DIR_GT_TEST].toString();

	m_zeroPadding = datasetConfig[ZERO_PADDING].toInt();

	checkAndCreateFolder(m_configPath + m_cleanTrainPath);
	checkAndCreateFolder(m_configPath + m_gtTrainPath);
	checkAndCreateFolder(m_configPath + m_cleanTestPath);
	checkAndCreateFolder(m_configPath + m_gtTestPath);

	m_startTrain = datasetConfig[START_TRAIN].toInt();
	m_stopTrain = datasetConfig[STOP_TRAIN].toInt();
	m_startTest = datasetConfig[START_TEST].toInt();
	m_stopTest = datasetConfig[STOP_TEST].toInt();

	#ifdef DEBUG
	Logger->debug("DataMemory::loadConfig() m_zeroPadding:{}", m_zeroPadding);
	Logger->debug("DataMemory::loadConfig() done");
	#endif
}

bool DataMemory::loadNamesOfFile()
{
	QVector<QString> m_imgList = scanAllImages(m_configPath+m_cleanPath);
	std::sort(m_imgList.begin(), m_imgList.end());
	#ifdef DEBUG
	Logger->debug("m_imgList:{}", m_imgList.size());
	#endif
	if (m_imgList.size() > 0)
	{
		int counterTrain{0};
		int counterTest{0};
		
		for (qint32 iteration = 0; iteration < m_imgList.size(); iteration++)
		{
			QString name = m_configPath + m_cleanTrainPath + m_split + m_imgList[iteration] + m_inputType;
			QString gt = m_configPath + m_gtTrainPath + m_split + m_imgList[iteration] + m_inputType;
			if (iteration % 100 == 0)
			{
				Logger->debug("DataMemory::loadNamesOfFile() file loading:{}/{}...", iteration, m_imgList.size());
				Logger->debug("DataMemory::loadNamesOfFile() name:{}", name.toStdString());
				Logger->debug("DataMemory::loadNamesOfFile() gt:{}", gt.toStdString());
			}
			if (iteration >= m_startTrain && iteration < m_stopTrain)
			{
				m_imageInfoTrain.push_back({ name.toStdString(), gt.toStdString() });
				counterTrain++;
			}
			else if (iteration >= m_startTest && iteration < m_stopTest)
			{
				m_imageInfoTest.push_back({ name.toStdString(), gt.toStdString() });
				counterTest++;
			}
		}
		Logger->info("DataMemory::loadNamesOfFile() file loaded:{}, train:{} test:{}", m_imgList.size(), counterTrain, counterTest);
	}
	else
	{
		Logger->warn("DataMemory::loadNamesOfFile() file non loaded{}");
		return false;
	}
	return true;
}

bool DataMemory::preprocess(QJsonArray dataGraph)
{
	#ifdef DEBUG_PREPROCESS
	Logger->debug("DataMemory::preprocess()");
	#endif
	m_cleanTrain.clear();
	m_cleanTest.clear();
	m_gtTrain.clear();
	m_gtTest.clear();
	m_data.clear();
	m_block.clear();
	m_graph = dataGraph;

	m_graph_processing.loadGraph(dataGraph, m_block);
	#ifdef DEBUG_PREPROCESS
	Logger->debug("DataMemory::preprocess() m_clean.size():{}",  m_clean.size());
	Logger->debug("DataMemory::preprocess() m_graph.size():{}",  m_graph.size());
	#endif
	
	for (qint32 iteration = 0; iteration < m_clean.size(); iteration++)
	{
		#ifdef DEBUG_PREPROCESS
		if (iteration % 100 == 0)
		{
			Logger->info("DataMemory::preprocess() processing:{}/{} m_clean.size():{}", iteration, m_clean.size(), m_inputData.size());
			Logger->info("DataMemory::preprocess() processing:{}/{} m_gtData.size():{}", iteration, m_gtData.size(), m_gtData.size());
		}
		#endif
		std::vector<cv::Mat> input{ m_clean[iteration], m_gt[iteration] };
		DataMemory::clearDataForNextIteration();
		
		// PROCESSING
		for (int i = 0; i < m_graph.size(); i++) 
		{
			std::vector<_data> dataVec;
			const QJsonObject _obj = m_graph[i].toObject();
			const QJsonArray _prevActive = _obj[PREV].toArray();
			const QJsonArray _nextActive = _obj[NEXT].toArray();
 
			if (m_graph_processing.checkIfLoadInputs(_prevActive, dataVec, input, i))
			{
				m_graph_processing.loadInputs(_prevActive, dataVec, dataGraph, m_data);				
			}
			try
			{
				m_block[i]->process((dataVec));
			}
			catch (cv::Exception& e)
			{
				const char* err_msg = e.what();
				qDebug() << "exception caught: " << err_msg;
			}
			m_data.push_back((dataVec));
			if (m_graph_processing.checkIfReturnData(_nextActive))
			{
				m_graph_processing.returnData(i, m_outputData, m_data);
			}
			dataVec.clear();
			#ifdef DEBUG_OPENCV
			if (m_outputData.size() > 0)
			{
				cv::imshow("m_outputData:", m_outputData[0]);
				cv::waitKey(0);
			}
			#endif
		}
		if (m_outputData.size() > 1)
		{
			if (iteration >= m_startTrain && iteration < m_stopTrain)
			{
				m_cleanTrain.push_back(m_outputData[0].clone());
				m_gtTrain.push_back(m_outputData[1].clone());
			}
			else if (iteration >= m_startTest && iteration < m_stopTest)
			{
				m_cleanTest.push_back(m_outputData[0].clone());
				m_gtTest.push_back(m_outputData[1].clone());
			}
		}
		#ifdef DEBUG
		Logger->debug("DataMemory::preprocess() iteration done");
		#endif
	}
	
	if(m_cleanTrain.size() != m_gtTrain.size())
	{
		Logger->error("DataMemory::preprocess() Train dataset size not correct");
	}
	if(m_cleanTest.size() != m_gtTest.size())
	{
		Logger->error("DataMemory::preprocess() Train dataset size not correct");
	}

	if(m_savePreprocessingDataset)
	{
		Logger->trace("DataMemory::preprocess() savePreprocessingDataset");
		for(int i = 0 ; i < m_cleanTrain.size() ; i++)
		{
			QString number = QString("%1").arg(i,m_zeroPadding, 10, QChar('0')); 
			QString pathToSaveClean = m_configPath + m_cleanTrainPath +  m_split + number + m_outputType;
			QString pathToSaveGt = m_configPath + m_gtTrainPath + m_split + number + m_outputType;
			Logger->trace("DataMemory::preprocess() pathToSaveClean:{}", pathToSaveClean.toStdString());
			Logger->trace("DataMemory::preprocess() pathToSaveGt:{}", pathToSaveGt.toStdString());
			cv::imwrite(pathToSaveClean.toStdString(), m_cleanTrain[i]);
			cv::imwrite(pathToSaveGt.toStdString(), m_gtTrain[i]);
		}
		for(int i = 0 ; i < m_cleanTest.size() ; i++)
		{
			QString number = QString("%1").arg(i, m_zeroPadding, 10, QChar('0')); 
			QString pathToSaveClean = m_configPath + m_cleanTestPath +  m_split + number + m_outputType;
			QString pathToSaveGt = m_configPath + m_gtTestPath + m_split + number + m_outputType;
			cv::imwrite(pathToSaveClean.toStdString(), m_cleanTest[i]);
			cv::imwrite(pathToSaveGt.toStdString(), m_gtTest[i]);
		}
	}

	#ifdef DEBUG
		Logger->debug("DataMemory::preprocess() sizes: m_clean:{}, m_gt{}", m_clean.size(), m_gt.size());
		Logger->debug("DataMemory::preprocess() sizes: m_cleanTrain.size:{}, ({}x{}) ", 
			m_cleanTrain.size(), m_cleanTrain[0].cols, m_cleanTrain[0].rows);
	#endif
	m_loaded = true;
	
	emit(memoryLoaded());
	return true;
}

void DataMemory::configure(QJsonObject const& a_config)
{
	#ifdef DEBUG
	Logger->debug("DataMemory::configure()");
	#endif
	DataMemory::loadConfig(a_config);

	
	m_loadData.configure(a_config);
	m_loadData.loadData(m_clean, m_gt);
	#ifdef DEBUG
	Logger->debug("DataMemory::configure() done");
	#endif
}
