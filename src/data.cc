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
constexpr auto WIDTH{ "Width" };
constexpr auto HEIGHT{ "Height" };

constexpr auto NOISE{"Noise"};
constexpr auto FOLDER{"Input"};

constexpr auto CLEAN{ "Clean" };
constexpr auto GT{ "Gt" };

constexpr auto STREAM_INPUT{"StreamInput"};
constexpr auto VIDEO_GT{"VideoGT"};
constexpr auto START_FRAME{"StartFrame"};
constexpr auto STOP_FRAME{"StopFrame"};
constexpr auto START_GT{"StartGT"};
constexpr auto STOP_GT{"StopGT"};
constexpr auto RESIZE{"Resize"};

constexpr auto PATH_TO_DATASET{ "PathToDataset" };
constexpr auto CONFIG_NAME{ "ConfigName" };

constexpr auto INPUT_TYPE{ "InputType" };
constexpr auto OUTPUT_TYPE{ "OutputType" };
constexpr auto INPUT_PREFIX{ "InputPrefix" };
constexpr auto DATASET_UNIX{ "DatasetLinux" };
constexpr auto DATASET_WIN32{ "DatasetWin32" };
constexpr auto CLEAN_TRAIN{ "Clean_train" };
constexpr auto GT_TRAIN{ "Gt_train" };

constexpr auto TRAIN_NUMBER{ "TrainNumber" };
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
	m_trainNumber = datasetConfig[TRAIN_NUMBER].toInt();
	checkAndCreateFolder(m_pathToConfig+m_cleanTrain);
	checkAndCreateFolder(m_pathToConfig+m_gtTrain);

	#ifdef DEBUG
	Logger->debug("DataMemory::loadConfig() done");
	#endif
}

bool DataMemory::loadNamesOfFile()
{
	QVector<QString> m_imgList = scanAllImages(m_pathToConfig+m_cleanTrain);
	std::sort(m_imgList.begin(), m_imgList.end());
	Logger->trace("m_imgList:{}", m_imgList.size());
	if (m_imgList.size() > 0) {
		for (qint32 iteration = 0; iteration < m_imgList.size(); iteration++) {
			if (iteration % 10000 == 0)
			{
				spdlog::info("iteration:{}", iteration);
			}

			QString name = m_pathToConfig + m_cleanTrain + m_split + m_imgList[iteration] + m_inputType;
			QString gt = m_pathToConfig + m_gtTrain + m_split + m_imgList[iteration] + m_inputType;
			if (iteration < m_trainNumber)
			{
				m_imageInfo.push_back({ name.toStdString(), gt.toStdString() });
			}
			
		}
	}
	return true;
}

bool DataMemory::preprocess(QJsonArray dataGraph)
{
	#ifdef DEBUG_PREPROCESS
	Logger->debug("DataMemory::preprocess()");
	#endif
	m_inputData.clear();
	m_gtData.clear();
	m_data.clear();
	m_block.clear();
	m_graph = dataGraph;

	m_graph_processing.loadGraph(dataGraph, m_block);
	#ifdef DEBUG_PREPROCESS
	Logger->debug("DataMemory::preprocess() m_cleanData.size():{}",  m_cleanData.size());
	Logger->debug("DataMemory::preprocess() m_graph.size():{}",  m_graph.size());
	#endif
	
	for (qint32 iteration = 0; iteration < m_cleanData.size(); iteration++)
	{
		#ifdef DEBUG_PREPROCESS
		if (iteration % 100 == 0)
		{
			Logger->info("DataMemory::preprocess() processing:{}/{} m_inputData.size():{}", iteration, m_cleanData.size(), m_inputData.size());
			Logger->info("DataMemory::preprocess() processing:{}/{} m_gtData.size():{}", iteration, m_gtData.size(), m_gtData.size());
		}
		#endif
		std::vector<cv::Mat> input{ m_cleanData[iteration], m_gtCleanData[iteration] };

		clearDataForNextIteration();
		
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
			Logger->debug("DataMemory::preprocess() push_back to m_inputData");
			if (m_outputData.size() > 0)
			{
				cv::imshow("m_outputData:", m_outputData[0]);
				cv::waitKey(0);
			}
			#endif
		}
		if (m_outputData.size() > 1)
		{
			m_inputData.push_back(m_outputData[0].clone());
			m_gtData.push_back(m_outputData[1].clone());
		}
		#ifdef DEBUG
		Logger->debug("DataMemory::preprocess() iteration done");
		#endif
	}

	#ifdef DEBUG
	Logger->debug("DataMemory::preprocess() sizes: m_cleanData:{}, m_gtCleanData{}", m_cleanData.size(), m_gtCleanData.size());
	Logger->debug("DataMemory::preprocess() sizes: m_inputData:{}, m_gtData{}", m_inputData.size(), m_gtData.size());
	#endif

	if(m_inputData.size() != m_cleanData.size())
	{
		Logger->error("DataMemory::preprocess() size not correct:");
	}
	Logger->debug("DataMemory::preprocess() sizes: m_inputData.size:{}, ({}x{}) ", m_inputData.size(), m_inputData[0].cols, m_inputData[0].rows);
	m_loaded = true;
	
	if(m_savePreprocessingDataset)
	{
		for(int i = 0 ; i < m_inputData.size() ; i++)
		{
			QString name_all = ( m_pathToConfig +m_cleanTrain +  m_split + QString::number(i) + m_outputType);
			Logger->debug("write:{}", (name_all).toStdString());
			cv::imwrite(name_all.toStdString(), m_inputData[i]);

			name_all = ( m_pathToConfig +m_gtTrain + m_split + QString::number(i) + m_outputType);
			Logger->debug("write:{}", (name_all).toStdString());
			cv::imwrite(name_all.toStdString(), m_gtData[i]);
		}
	}
		
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
	m_loadData.loadData(m_cleanData, m_gtCleanData);
	#ifdef DEBUG
	Logger->debug("DataMemory::configure() done");
	#endif
}
