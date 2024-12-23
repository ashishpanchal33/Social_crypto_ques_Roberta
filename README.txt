TABLE OF CONTENT
------------------------------------------------------------
1. DESCRIPTION
- 1.1 List of modules
- 1.2 Data File structure
2. INSTALLATION
3. EXECUTION
- 3.1 Python module suit
- 3.2 Refresh Tableau Dashboard
- 3.3 Run Dashboard Final.twb in Tableau Desktop, and refresh the data.
- 3.4 Visualize and make your investment decisions


VIZ URL : https://team93-group-project.herokuapp.com/

--------------------------------------------------------------
--------------------------------------------------------------




1. DESCRIPTION - MODULES and ADS

Social crypto cues is employees 17 modules, working together to create the.
The details of the module is added below


1.1 List of modules:
	|---1	social_crypto_cues.py
								: Main project module, used to invoke other modules as per requirements.
								- modules 2,11,12,13,14,15,16 , labeled as [1,2,3,4,5,6,7]
								can be invoked in a serial order with the help of arguments, seperated by space.
								------- example : python social_crypto_cues.py 1 2 3
										This would run module 2, 11 and 12.
								If any incorrect process/module number is provided in the argument, it is skipped to process remaining, will a flash error. 
								
								In case no arguments are provided, all the processes are invoked in a serial order.
								------- example : python social_crypto_cues.py
						
	|---2	Main_social_listening_and_processing_script.py
								: Main social listening module used for invoking the complete social listening process.
								- note, unlike social_crypto_cues.py, all the processes are invoked.
								
								- defined parameters:
									- subreddit_list = ['CryptoCurrency','CryptoMarkets']  : subreddits to scrape data from
									- duration = 31 : number of days for batch processing - reddit scrapping
									- post_limit =200 : number of post to be crawled per day
									- start = (2022, 3, 6) # start day of scrapping
									- end = (2022,4,22)   # end day of scrapping ( has to be more than start + duration, else the data for start+duration is auto scrapped)
									
								- fixed parameter :
									- column_list : list of factors to be considered for popularity and hotness calculations.
									
								- process :
									a. extract reddit posts : functions from module 4 : Get_post_data.py (get_post_data_and_summary)
										- input: (defined parameters)
										- output: day and subreddit level post data along with meta data saved in './redditdata/' 
											- seperator : "|"
											- file(s) name format -  subreddit+date.strftime("%d_%m_%y")+".txt"
											
										
									b. identify day level topics : function from module 5: Post_Token_identifications.py (generate_Data)
										- input : single post data frame for the time period, from './redditdata/'
										- output: identified daily tokens (converstaion themes from posts) and associated daily metrics, 
											- output location : './reddit_post_summary_month'
											- file format : csv
											- file name format : "trend_data_"+month+".csv"
											
									c: calculate sentiments and emotions per post for provided posts data :  function from module 7: sentiment_analysis.py (run_onlist_loop)
										- input : single post data frame for the time period, from './redditdata/'
										- output: identified sentiment and emotion score per posts, 
											- output location : './redditdata_sentiment'
											- file format : txt, seperated by :"|"
											- file name format : "date_"+".txt"
									
									d: token level sentiment calculations :  function from module 8: generate_token_level_sentiment.py (run_sentiment_agg)
										- input : singe token data frame from './reddit_post_summary_month/', and single data frame from  './redditdata_sentiment/'
										- output: daily token level sentiments aggregations along with existing token metadata, 
											- output location : './reddit_post_summary_day/'
											- file format : csv
											- file name format : "trend_data_"+date_+".csv"										
									e: Calculating rolling popularity and hotness for requisite factors :  function from module 9: calculate_pop_hot.py (calculate_popularity_hotness_rising_)
										- input : singe token with sentiment frame from './reddit_post_summary_day/', and column_list : name of columns for rolling score.
										- output: rolling scores ( popularity and hotness) for factors assocaited with a token per day, 
											- output location : './reddit_post_score_summary_month/'
											- file format : csv
											- file name format : "all_summary_"+month+".csv"	
									
									
									
									f: Creating connection nodes and edges and calculating connection strength:  function from module 10: create_post_connections.py (get_connections)
										- input : daily token with sentiment and score data frame from './reddit_post_score_summary_month/',
										- output: day level connections of top token with connetion strength and selected top token and assocaited token popularity data, ( for top 300 tokens wper day ) 
											- output location : './reddit_topic_connections_day/' : connections and connection strengths
											'./reddit_topic_weights_connections_day/' : selected token metadata.
											- file format : csv
											- file name format : date__+".csv"										
									
	|----..........3	Import_lib.py:
									- based file to import all social listening process libraries
									(some libraries are directly callind in individual modules)
	|----..........4	Get_post_data.py:
									- download reddit data using reddit API and Pushift API.
									- important definable parameters:
										- reddit_read_only,reddit_authorized : Reddit session object: for reddit data authentication : used for extracting additional post metadata.
										
										-api = PushshiftAPI() : pushift API object : used to extract historical post data
									-important functions : 
									a. get_post_data_and_summary
										- extraction period~duration quantizer
									b. bind_data_overdate:
										- day level extraction chunker and envokations
									
									c. get_data	: day level reddit API envoker - scrapper, post metadata extractor and data post data storing.
									
									- output locations : refer module 2.
									
									
	|----..........5	Post_Token_identifications.py
									- 30000 token identifications from post data per data
									- type: unigram, bigram
									-using : Count vectorizer
									- Also create aggregated fields for token like: weight: number of posts
											'ups'
											,'downs'
											,'avg_upvote'
											,'num_comments'
											,'score_total'
											,'total_awards'
											,'post_id' 
											,'Flair_list'
											,'daily_popularity'
											,'noun' : validate if token is a noun
											
									-functions :
										a. generate_Data : monthly level process chunker ( batch processing)
										b. month_summary : month level data concat and ay level process envoker
										c. create_summary_each_day : data level token identifications and processing.
										
									- output locations : refer module 2.
										
	|----..........6	get_data_frames.py
									- general data fetch module to create single files from multiple files from the data structure.
									-uses - glob	
	
	
	|----..........7	sentiment_analysis.py
									a. initlize sentiment modules:
										-functions : ( per engine)
											roberta_init
											vader_init
											distilbert_emo_init
									b. calculate sentiment and emotions of per post:
										- functions:
											-- get_emo_df : date level process chunker.
											-- get_emotions: calculate emotions using roberta, vader and distillbert for each post per day.
													
											-- roberta_sent : calulate sentiment scores using roberta per text+title
											-- vader_sent : calulate sentiment scores using vader per text+title
											-- distilbert_emo : calulate emotion scores using distilbert per text+title
											
											
									- output locations : refer module 2
										
										
										
										
										
	|----..........8	generate_token_level_sentiment.py
									: calculate sentiment aggregations per token at a day level :i.e 
										- sum
										- weighted sum (on popularity)
										- mean
									-functions : 
										run_sentiment_agg : date level process chunker
										sentiment_agg : token level data aggrigator
										wt_sum : sentiment weighted score calulation 

											
									- output locations : refer module 2	
	
	
	
	
	
	
	
	
	
	
	|----..........9	calculate_pop_hot.py
	
									: calculate factors rolling aggregations : popularity and hotness 
									-functions : 
										-calculate_popularity_hotness_rising_ : month and day level process chunker

										-run_for_one : run for 1 day - calculate  (-1) month date delta from current day, extract relevant 1 month data frame for processing and envoke pop_assist per token per day.
										-pop_assist_2 : calulcate rolling 30 days average popularity ( i.e sum divided by 30)
										and rolling 3 days average popularity ( ie. divide by 4*3)

											
									- output locations : refer module 2		
	
	
	
	
	
	
	
	
	
	|----..........10	create_post_connections.py:
									: Create post token daily connections nodes and edges dataframes per day
									- functions:
										- get_connections : day level process chunker
										- connection_per_day : daily top 50 + token identified and relative top 6 connection token identification invoker 
										- find_top_6_connections:
											- daily~per token - top 6 associated tokens identified (based on rolling 30 days popularities) and connection weight calulator - number of common posts.
											
									- output locations : refer module 2		
										
	|
	|---11	fetch_price_info_and_fundamental_factors.py:
									data extraction module for fundamental and technical factors:
									- functions:
										--- get_hist_data : historical crypto price data extractor for given period: uses cryptocompare data API ( no configurlation required)
										--- data_to_dataframe: convert dict to pandas dataframe
										--- dw_get : base API function/query for Data world data extraction
										--- scrap_crypto_price_data: price data extraction batch processor per coin for given period.
											-- target_currency set to = "USD"
											
										--- scrapping_search_data:
											-- search trend data extractor (google trends API) for top 100 crypto coins, as per crypto market cap, fetched from dataworld API.
										--- scrap_external_factors:
											extract macroeconomic factors, stock price and market index ['AMD','NVDA,'^DJI','^GSPC','^IXIC','^NYA','gold','RETAIL_SALES','CONSUMER_SENTIMENT','UNEMPLOYMENT']
										--- main_trends_data:
											main extraction invoking functions, combining data from different sources.
											- important params:
												- coinlist : list of coins for price data extraction
												-    start_ =(2019, 6, 22): start date for extraction
												-    end_ = (2022, 4,17) : end date extraction.
												
												
										- output :  'input/dataset.csv'
										- output factors : "high,low,open,volumefrom,volumeto,close,conversionType,sym,date,SearchFrequency,tags,amd_open,nvda_open,retail_sales,dji_open,gspc_open,ixic_open,nya_open,gold"
	
	
	
	
	|---12	clustering.py:
										- extract crypto meta data from dataworld and create tag based crypto currency clusters for tableau dashboard screen 1, for 100 crypto using kmodes
										- extract relevant wikipedia data for 100 crypto
										
										-functions:
											a. main_coin_clustering_and_data_proc:
												main module, invoking other functions.
												- aggregating and concatinating data modules.
											b. get_cleaned_dw_data:
												- extract tag data for top 100 crypto currencies from data world and cleaning tags including removing "portfolio tags"
											c. dw_get : base method to get data from data world crypto table
											d. create_record: convert list to dict record
											e. clustering_elbow_curve : kmodes clustering module to calculate the best cluster splits, by identifying best number of clusters.:
												- supports manual mode - where the plot of k_range can be visualized and k can be identified. ( currecntly k_range is set to 10-30,
												- auto mode: utlizeses silhouette method to identify the best k ( may not perform well)
											f. cluster_data : finally cluster coins based on bset "k" ( currently hard fixed to 20
											g. get_wikipedia_data:
												extract wikiperida data using wikipedia search api, with different versions of crypto currency idetnifier tags.
											
										-output : ( refer data structure for details )
											_ 'input/wiki_info.csv'
											a. 'input/cluster_info.csv'
											b. 'output/about_data.csv'
											c. 'output/cluster_info.csv'
											
										
										
	
	
	|---13	Price_prediction_xgboost.py:
										- main price prediction module using fundamental, technical and social factors
										
										-functions:
											------ get_pred_frame(): get single data frame from the given location containing csvs default value = (./reddit_post_score_summary_month/)
											------ get_token_list(): get cluster info data from "input/cluster_info.csv
											------ sentiment_factors(): fetch the names of all possibele sentiment factors.
											------ fetch_and_create_base_data(): cuntion used to create base data for model predictions , including 'input/dataset.csv' and 'input/symbol_name_mapping.csv'
													- output : sym_pop_data - sentiment data, data - price and other factor data.
											------ make_model(df1,model,ratio,rp=1,param=None): xgboost base model function : for testing and final predictions
												- utlizes XGBRegressor, with hyper parametrs found on gridsearch.
											------ print_stats(mine,df,x,y):
												- module which canculate hte association score of two factors: currently recording MIC,GMIC and TIC.
											------ default(obj): float data handler for json.
											------ smooth_data (df1): savgol data smoothning filter. for all columns besides fundamental factors
													- degree = 3, number of days relateive smoothning  = 30
											------ filter_and_scale(df,sym,date,param, verbose=0): minmaxscaler for given columns of the dataframe.
											------ shift_by_n(df,n,delta = 1):
													model data horizon shifter.
											------ apply_ta_features(df_scaled):
													- function to create technical factors based on fundamental factors.
											------ main_grid_search_and_prediction_process():
												- main process modules
												- run process for each coin persent in data ( from fetch_and_create_base_data())
												- create additional sentiment factors and connect price and sentiment factors
												- invoke ta feature caculate model
												- invoke horizon shifter by 30 days
												- invoke scaling of data.
												- smoothen the data
												- identify viable time range of data for the given coin, with the first encounter of coin price >= (average close price)/2.3
												- store unscaled data at : 'output/unscale_ata/'
												- store scalled data at: 'output/scale_ata/' 
												- identify factors with gmin more than dynamic thresholds, adjusted to identify atleast 5 factors for training and prediction.
												-store GMIC values for symbol at : "output/associate_factors_df/"
												- store pattern of selected factors at 'output/associate_factors_html/'+sym+'.html'
												- perform grid search:
													- define grid search ranges
													- define gridsearch object with params 
													- perform grid search asn store the gridsearch performance data at : "output/grid_results/"+sym+".json"
													- find best result of grid search per coin and save at "output/grid_results_best_param/"+sym+".csv"
												- perform xgboost testing based on best grid search params on latest 5% data of the viable time range:
													- save error score of best param for testing at : "output/pre_best/"+sym+".csv"
													- join and inverse scale prediction data:
													- store predictions and prediction comparisons at : "output/model_predictions/"+sym+".csv" and "output/model_predictions_comparison/"+sym+".csv" respectively
													- save test prediction chart at : 'output/prediction_html/'+sym+'.html'
													
												- final forecasting:
													- predict for future 30 days from the last date in the data.
													- save score  from 0 value at "output/pre_best_30_days/"+sym+".csv"
													- save predictions and comparison with 0 value at "output/model_predictions_30_days/"+sym+".csv" and "output/model_predictions_comparison_30_days/"+sym+".csv" respectively 
													- store prediction chart at : 'output/prediction_html_30_days/'+sym+'.html'
													
													
											
											
	|---14	Create_tableau_required_trends_data.py:
										- Create consolidated data for tablue inclusing :
											a. single file for Scaled ta factors
											b. single file for Scaled social factors
											c. single file for model forecast results
											d. future close price and factors correlation ( MIC,GMIC,TIC)	
											
										-functions:
											-- get_frame : modele to read all csv data from a directory and create single data frame, with an option to sort data.
											-- factors_lookup: list of sentiment and fundamental factors
											-- create_combined_scaled_ta_data : create single data files for:
												- technical and fundamental indicators : './output/scaled_dataset.csv'
												- social sentiment factors :  './input/reddit_pop_data.csv'
											-- create_combined_results_data: create single file for xgboost forecasting results: './output/results.csv'
											-- create_correlation_data : combining and creating a single file for visualizing correlation of coin future price and factors: './output/correlation.csv'
	|---15	Network_builder.py:
										Generate Tableau screen 1 network data using networkx:
										-functions
											a. get_connection_frame: data fetch function with single file data frame creation : default to fetch data from './reddit_topic_connections_day/'
											b. create_nodes:
												- create filtered tokens based on given minimum date, from  './reddit_topic_connections_day/': 'input/filtered_tokens.csv'
												- create base_node_file for tokens with given date filter from './reddit_topic_weights_connections_day/' : 'input/filtered_tokens_nodes.csv'
												
											c. create_clusters: base function for creating network chart for the selected nodes, ( calculating best x,y location of the nodes, mapped to their associated connections along with connections strngh and individual node importance : 'output/nw_main.csv'
	|---16	create_post_text_tableau_data.py:
										Generate post data file for tableau screen 1
										-functions : 
												- get_post_text_data : fetch all post data with sentiments as a single dataframe. : './redditdata_sentiment/'
												- get_connection_frame : general method to fetch network connection data from './reddit_topic_connections_day/'
												- main_post_text_tableau_function:
														-	 main method invoking post text fetch, and connection fetch, post a speficied date
														-   process data for tableau dashboard including cleaning, mapping and reddit url generation
														- output : 'output/post_text.csv'
										
	|---17	post_tokens_scoring_script.py : Not in use




1.2 Data File structure
	|---1'./input',
	|		 |---- './cluster_info.csv', : cryptocurrency name and cluster information used by Screen 1 in Tableau dashboard
	|		 |---- './dataset.csv', : all price data information for 20 currencies considered for modelling. Used by prediction model as input
	|		 |---- './filtered_tokens.csv', : token association data used to generate network graph coordinates used in Screen 1 of Tableau dashboard
	|		 |---- './filtered_tokens_nodes.csv', token wise daily poularity data used to generate network graph used in Screen 1 of Tableau dashboard
	|		 |---- './reddit_pop_data.csv', day wise popularity data, used to for prediction model as well as Screen 2 of Tableau dashboard
	|		 |---- './symbol_name_mapping.csv', : static
	|		 |---- './wiki_info.csv', : intermediate file
	|
	|
	|
	|---2'./minepy-1.2.5', : incase you are not able to install minepy through pip or conda forge, you may downloand and install through whl built package from : https://www.lfd.uci.edu/~gohlke/pythonlibs/#minepy, supporting your environment.
	|---3'./output',
	|		 |---- './about_data.csv', : token wise information extracted from wikipedia used by Screen 1 in Tableau dashboard
	|		 |---- './cluster_info.csv',: cryptocurrency name and cluster information used by Screen 1 in Tableau dashboard
	|		 |---- './correlation.csv', : correlation coefficients of all factors used of cryptocurrency predictions used by Screen 2 in Tableau dashboard
	|		 |---- './results.csv', : prediction results generated by XGBoost model used by Screen 2 in Tableau dashboard
	|		 |---- './nw_main.csv', : Token wise network coordinates generated to show association between tokens used by Screen 1 in Tableau dashboard
	|		 |---- './post_text.csv', Token wise Reddit posts extracted used by Screen 1 in Tableau dashboard
	|		 |---- './scaled_dataset.csv', Scaled technical and fundamental factors for all 20 cryptocurrencies. Also used by Screen 2 in Tableau dashboard
	|		 |
	|		 |************ Prediction sub-file Data-structure ****
	|		 |---- './associate_factors_df', : contains "coin-symbol".csv includes correlation parameterers MIC, GMIC, TIC between all cryptocurrencies and their associated factors.
	|		 |---- './associate_factors_html', : contains "coin-symbol".html includes graphs generated of price Vs associated factors
	|		 |---- './grid_results' contains "coin-symbol".json includes grid search results for all cryptocurrencies
	|		 |---- './grid_results_best_param', : contains "coin-symbol".csv includes best factors determined by grid search
	|		 |---- './model_predictions' , : contains "coin-symbol".csv includes prediction results
	|		 |---- './model_predictions_30_days', : contains "coin-symbol".csv includes predictor variables and value for all cryptocurrencies
	|		 |---- './model_predictions_comparison', : contains "coin-symbol".csv includes predictor variable and value while training model 
	|		 |---- './model_predictions_comparison_30_days', contains "coin-symbol".csv includes predicted and actual value while training
	|		 |---- './pre_best', : Scores for best hyper parameters
	|		 |---- './pre_best_30_days', : Scores for best hyper parameters
	|		 |---- './prediction_html', : contains "coin-symbol".html and stores predicted graph
	|		 |---- './prediction_html_30_days', : contains "coin-symbol".html and stores predicted graph by training
	|		 |---- './scale_ata', contains scaled attributes used for prediction
	|		 |---- './unscale_ata' contrains unscaled attributes used for prediction
	|
	|**********Social listening processed data-structure********
	|
	|---4'./reddit_post_score_summary_month', : refer module 2 for details
	|---5'./reddit_post_summary_day', : refer module 2 for details
	|---6'./reddit_post_summary_month', : refer module 2 for details
	|---7'./reddit_topic_connections_day', : refer module 2 for details
	|---8'./reddit_topic_weights_connections_day', : refer module 2 for details
	|---9'./redditdata', : refer module 2 for details
	|--10'./redditdata_sentiment' : refer module 2 for details














2. INSTALLATION

- setup a new python environment : refer ( conda : https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html, virtualenv : https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) 
- Install packages from requirements.txt (pip install -r /path/to/requirements.txt)
- Setting up data APIs:
	1. Search Frequency data (using pytrends) -- Login to Google once to fetch results
	2. Data World (using dw) -- 1. Login to data.world
					-- 2. go to settings and "enable" python integration 
						  -- 3. Request for token
					-- 4. run dw configure on terminal and provide the access token generated.
	3. (no api needed) Data is generated as jsons from 'https://www.alphavantage.co/query?'
	4. (no api needed) pandas_datareader is used to fetch data from Yahoo Financial!
	5. REDDIT API:
		-Install the praw package: $ pip install praw
		-In a text editor, create a client_secrets.json file with that structure.

		{
			"client_id":"",
			"client_secret":"",
			"user_agent":"script by u/yourusername",
			"redirect_uri":"http://localhost:8080",
			"refresh_token":""
		}

		-Now create an application to query the API.

		1. Go to Reddit Apps
		2. Select “script” as the type of app.
		3. Name your app and give it a description.
		4. Set-up the redirect uri to be http://localhost:8080.
		5. The redirect URI will be used to get your refresh token.

		-Once you click on “create app”, you will get a box showing you your client_id and client_secrets. Copy those values in your client_secrets.json file.

		-It is now time to generate a refresh token.The refresh token will be useful to access the API without always re-approving the API.Provide client-id and client-secret when generating token. Set scope to "all". Now, copy the url in browser and authorize the app. You will be redirected to a page where you access token is printed. Copy the refresh token into your client_secrets.json file.
		
		- you may directly copy the generated client_id, client_secret and updated details in "Get_post_data.py":	reddit_read_only,reddit_authorized 


- copy the CODE folder to the python environment accessible directory, unless dynamic access is enabled.

- Install Tableau Desktop : https://help.tableau.com/current/desktopdeploy/en-us/desktop_deploy_download_and_install.htm



3. EXECUTION

3.1 Python module suit: there are 2 versions provided to the ease of utility and testing:
	|
	|--- a. From Scratch : without any additional data, where the tester can start from extraction of data:
						- total time taken by data extraction and processing may exceed 30~40 hours, however as the data is serially extracted the process can be stopped and restarted from a checkpoint by exclusively adding date range changes in the Get_post_data.py file.
						
						- "From Scratch" version directory is provided in "CODE" directory
						- to execute this process you may invoke the following command, after navigating inside "From Scratch" directory from your python console
							- To run all processes in series : python social_crypto_cues.py
	|
	|--- b. With Data	 : Data pack is provided to enable the user to walkthrough the existing data.
						- post extraction, the user may test and execute any invokation command, with step skips.
						- example of skipping reddit data extraction: python social_crypto_cues.py 2 3 4 5 6 7
						- example of only to fetch_price_info_and_fundamental_factors : python social_crypto_cues.py 2

							  
	-- please refer Section 1.1.1 DESCRIPTION>List of modules>social_crypto_cues.py for more details on execution command.
						


3.2 Refresh Tableau Dashboard
	- Dashboard Final.twb is provided in both "From Scratch" and "With Data"
	- to use the same you would need to refresh the data post excecution.
	To refresh tableau dashboard following files generate by the python script are required. 

	- output/about_data.csv (Required for sheet "about" to generate currency level text information)
	- output/cluster_info.csv (Required for sheet "clusters" to generate associated clusters and shared characteristics)
	- output/post_text.csv (Required for sheet "post_text" to generate Reddit text data associated with selected token)
	- output/nw_main.csv (Required for sheet "network" to generate coordinates for network graph generated on token association and popularity)
	- output/correlation.csv (Required for sheet "correlation" to generated correlation coefficients of price with different variables)
	- output/results.csv (Required for sheet "Predicted" to generate final predictions for all cryptocurrencies)
	- input/reddit_pop_data.csv (Required for sheet "Sentiment" to generate day wise sentiment score derived from Reddit text data)
	- output/scaled_dataset.csv (Required for sheet "Trends" to understand fundamental and technical factor dependencies)



3.3 Run Dashboard Final.twb in Tableau Desktop, and refresh the data.
3.4 Visualize and make your investment decisions


