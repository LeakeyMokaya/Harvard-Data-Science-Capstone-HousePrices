#' ---
#' title: 'HarvardX Data Science Capstone: House Prices Report'
#' author: "Justin Nielson"
#' date: "May 30, 2019"
#' output:
#'   word_document:
#'     toc: yes
#'   pdf_document:
#'     latex_engine: xelatex
#'     toc: yespur
#' ---
#' 
## ----setup, include=FALSE------------------------------------------------
knitr::opts_chunk$set(
	echo = TRUE,
	message = FALSE,
	warning = FALSE
)

#' 
#' ## 1. Introduction 
#' 
#' The HarvardX Data Science Capstone House Prices project is an ongoing Kaggle competition using advanced regression techniques to predict sales prices and practice feature engineering, RFs, and gradient boosting. The link to the competition is here. https://www.kaggle.com/c/house-prices-advanced-regression-techniques
#' 
#' I thought this would be a good choose your own project in that it builds on the linear regression methods that were used in the MovieLens movie recommendation system project. The competition also allows me to get feedback from the Kaggle community on my methods, analysis, and results. 
#' 
#' The submissions are evaluated on Root Mean Squared Logarithmic Error (RMSLE) between the logarithm of the predicted value and the logarithm of the observed sales price. The objective is for each Id in the test set, the model must predict the value of the SalePrice variable and minimize the RMSLE. 
#' 
#' The ML model used with the smallest RMSLE was the *Lasso regression model* on the training set with a RMSLE of 0.0978573.
#' 
#' ## 2. Overview
#' 
#' This report contains sections for data exploration, visualization, preprocessing, evaluated machine learning algorithms, and RMSLE analysis sections including methods that were used to transform the data to create the best predicive model.
#' 
#' The results and conclusion sections and the end includes final thoughts on the House Prices project.
#' 
#' ###   2.1. Loading libraries and data
## ----message=FALSE, warning=FALSE----------------------------------------
# Loading packages for data exploration, visualization, preprocessing, 
# machine learning algorithms, and RMSLE analysis

library(tidyverse)
library(caTools)
library(caret)
library(e1071)
library(glmnet)
library(randomForest)
library(xgboost)
library(data.table)
library(lubridate)
library(ggplot2)
library(corrplot)
library(knitr)
library(kableExtra)

# Read House Prices train dataset:
train <- read.csv("train.csv")

# Fill NA with 0 for modeling purposes
Train <- train %>% mutate_all(~replace(., is.na(.), 0))

# Read House Prices test dataset:
test <- read.csv("test.csv")

# Fill NA with 0 for modeling purposes
test <- test %>% mutate_all(~replace(., is.na(.), 0))

#Converting character variables to numeric

train$paved[train$Street == "Pave"] <- 1
train$paved[train$Street != "Pave"] <- 0

train$regshape[train$LotShape == "Reg"] <- 1
train$regshape[train$LotShape != "Reg"] <- 0


#' 
#' ## 3. Executive Summary
#' 
#' The House Prices Kaggle data includes four files train.csv, test.csv, sample_submission.csv and data_description.txt. The train dataset includes 1,459 objects or rows of 81 variables were the test dataset includes 1,460 objects or rows of 80 variables. The last variable in the train dataset is the actual SalesPrice that we are trying to predict in the test dataset. The samp_submission file format is the Id of the house and predicted SalesPrice. The data_description.txt file includes descriptions of all the variables in the train and test datasets. 
#' 
#' *House Prices train dataset*
#' 
## ----echo=TRUE, message=FALSE, warning=FALSE-----------------------------

summary(train)


#' 
#' *House Prices test dataset*
#' 
## ----echo=TRUE, message=FALSE, warning=FALSE-----------------------------

summary(test)


#' 
#' *House Prices Data Description*
#' 
#' MSSubClass: Identifies the type of dwelling involved in the sale.	
#' 
#'         20	1-STORY 1946 & NEWER ALL STYLES
#'         30	1-STORY 1945 & OLDER
#'         40	1-STORY W/FINISHED ATTIC ALL AGES
#'         45	1-1/2 STORY - UNFINISHED ALL AGES
#'         50	1-1/2 STORY FINISHED ALL AGES
#'         60	2-STORY 1946 & NEWER
#'         70	2-STORY 1945 & OLDER
#'         75	2-1/2 STORY ALL AGES
#'         80	SPLIT OR MULTI-LEVEL
#'         85	SPLIT FOYER
#'         90	DUPLEX - ALL STYLES AND AGES
#'        120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
#'        150	1-1/2 STORY PUD - ALL AGES
#'        160	2-STORY PUD - 1946 & NEWER
#'        180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
#'        190	2 FAMILY CONVERSION - ALL STYLES AND AGES
#' 
#' MSZoning: Identifies the general zoning classification of the sale.
#' 		
#'        A	Agriculture
#'        C	Commercial
#'        FV	Floating Village Residential
#'        I	Industrial
#'        RH	Residential High Density
#'        RL	Residential Low Density
#'        RP	Residential Low Density Park 
#'        RM	Residential Medium Density
#' 	
#' LotFrontage: Linear feet of street connected to property
#' 
#' LotArea: Lot size in square feet
#' 
#' Street: Type of road access to property
#' 
#'        Grvl	Gravel	
#'        Pave	Paved
#'        	
#' Alley: Type of alley access to property
#' 
#'        Grvl	Gravel
#'        Pave	Paved
#'        NA 	No alley access
#' 		
#' LotShape: General shape of property
#' 
#'        Reg	Regular	
#'        IR1	Slightly irregular
#'        IR2	Moderately Irregular
#'        IR3	Irregular
#'        
#' LandContour: Flatness of the property
#' 
#'        Lvl	Near Flat/Level	
#'        Bnk	Banked - Quick and significant rise from street grade to building
#'        HLS	Hillside - Significant slope from side to side
#'        Low	Depression
#' 		
#' Utilities: Type of utilities available
#' 		
#'        AllPub	All public Utilities (E,G,W,& S)	
#'        NoSewr	Electricity, Gas, and Water (Septic Tank)
#'        NoSeWa	Electricity and Gas Only
#'        ELO	Electricity only	
#' 	
#' LotConfig: Lot configuration
#' 
#'        Inside	Inside lot
#'        Corner	Corner lot
#'        CulDSac	Cul-de-sac
#'        FR2	Frontage on 2 sides of property
#'        FR3	Frontage on 3 sides of property
#' 	
#' LandSlope: Slope of property
#' 		
#'        Gtl	Gentle slope
#'        Mod	Moderate Slope	
#'        Sev	Severe Slope
#' 	
#' Neighborhood: Physical locations within Ames city limits
#' 
#'        Blmngtn	Bloomington Heights
#'        Blueste	Bluestem
#'        BrDale	Briardale
#'        BrkSide	Brookside
#'        ClearCr	Clear Creek
#'        CollgCr	College Creek
#'        Crawfor	Crawford
#'        Edwards	Edwards
#'        Gilbert	Gilbert
#'        IDOTRR	Iowa DOT and Rail Road
#'        MeadowV	Meadow Village
#'        Mitchel	Mitchell
#'        Names	North Ames
#'        NoRidge	Northridge
#'        NPkVill	Northpark Villa
#'        NridgHt	Northridge Heights
#'        NWAmes	Northwest Ames
#'        OldTown	Old Town
#'        SWISU	South & West of Iowa State University
#'        Sawyer	Sawyer
#'        SawyerW	Sawyer West
#'        Somerst	Somerset
#'        StoneBr	Stone Brook
#'        Timber	Timberland
#'        Veenker	Veenker
#' 			
#' Condition1: Proximity to various conditions
#' 	
#'        Artery	Adjacent to arterial street
#'        Feedr	Adjacent to feeder street	
#'        Norm	Normal	
#'        RRNn	Within 200' of North-South Railroad
#'        RRAn	Adjacent to North-South Railroad
#'        PosN	Near positive off-site feature--park, greenbelt, etc.
#'        PosA	Adjacent to postive off-site feature
#'        RRNe	Within 200' of East-West Railroad
#'        RRAe	Adjacent to East-West Railroad
#' 	
#' Condition2: Proximity to various conditions (if more than one is present)
#' 		
#'        Artery	Adjacent to arterial street
#'        Feedr	Adjacent to feeder street	
#'        Norm	Normal	
#'        RRNn	Within 200' of North-South Railroad
#'        RRAn	Adjacent to North-South Railroad
#'        PosN	Near positive off-site feature--park, greenbelt, etc.
#'        PosA	Adjacent to postive off-site feature
#'        RRNe	Within 200' of East-West Railroad
#'        RRAe	Adjacent to East-West Railroad
#' 	
#' BldgType: Type of dwelling
#' 		
#'        1Fam	Single-family Detached	
#'        2FmCon	Two-family Conversion; originally built as one-family dwelling
#'        Duplx	Duplex
#'        TwnhsE	Townhouse End Unit
#'        TwnhsI	Townhouse Inside Unit
#' 	
#' HouseStyle: Style of dwelling
#' 	
#'        1Story	One story
#'        1.5Fin	One and one-half story: 2nd level finished
#'        1.5Unf	One and one-half story: 2nd level unfinished
#'        2Story	Two story
#'        2.5Fin	Two and one-half story: 2nd level finished
#'        2.5Unf	Two and one-half story: 2nd level unfinished
#'        SFoyer	Split Foyer
#'        SLvl	Split Level
#' 	
#' OverallQual: Rates the overall material and finish of the house
#' 
#'        10	Very Excellent
#'        9	Excellent
#'        8	Very Good
#'        7	Good
#'        6	Above Average
#'        5	Average
#'        4	Below Average
#'        3	Fair
#'        2	Poor
#'        1	Very Poor
#' 	
#' OverallCond: Rates the overall condition of the house
#' 
#'        10	Very Excellent
#'        9	Excellent
#'        8	Very Good
#'        7	Good
#'        6	Above Average	
#'        5	Average
#'        4	Below Average	
#'        3	Fair
#'        2	Poor
#'        1	Very Poor
#' 		
#' YearBuilt: Original construction date
#' 
#' YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
#' 
#' RoofStyle: Type of roof
#' 
#'        Flat	Flat
#'        Gable	Gable
#'        Gambrel	Gabrel (Barn)
#'        Hip	Hip
#'        Mansard	Mansard
#'        Shed	Shed
#' 		
#' RoofMatl: Roof material
#' 
#'        ClyTile	Clay or Tile
#'        CompShg	Standard (Composite) Shingle
#'        Membran	Membrane
#'        Metal	Metal
#'        Roll	Roll
#'        Tar&Grv	Gravel & Tar
#'        WdShake	Wood Shakes
#'        WdShngl	Wood Shingles
#' 		
#' Exterior1st: Exterior covering on house
#' 
#'        AsbShng	Asbestos Shingles
#'        AsphShn	Asphalt Shingles
#'        BrkComm	Brick Common
#'        BrkFace	Brick Face
#'        CBlock	Cinder Block
#'        CemntBd	Cement Board
#'        HdBoard	Hard Board
#'        ImStucc	Imitation Stucco
#'        MetalSd	Metal Siding
#'        Other	Other
#'        Plywood	Plywood
#'        PreCast	PreCast	
#'        Stone	Stone
#'        Stucco	Stucco
#'        VinylSd	Vinyl Siding
#'        Wd Sdng	Wood Siding
#'        WdShing	Wood Shingles
#' 	
#' Exterior2nd: Exterior covering on house (if more than one material)
#' 
#'        AsbShng	Asbestos Shingles
#'        AsphShn	Asphalt Shingles
#'        BrkComm	Brick Common
#'        BrkFace	Brick Face
#'        CBlock	Cinder Block
#'        CemntBd	Cement Board
#'        HdBoard	Hard Board
#'        ImStucc	Imitation Stucco
#'        MetalSd	Metal Siding
#'        Other	Other
#'        Plywood	Plywood
#'        PreCast	PreCast
#'        Stone	Stone
#'        Stucco	Stucco
#'        VinylSd	Vinyl Siding
#'        Wd Sdng	Wood Siding
#'        WdShing	Wood Shingles
#' 	
#' MasVnrType: Masonry veneer type
#' 
#'        BrkCmn	Brick Common
#'        BrkFace	Brick Face
#'        CBlock	Cinder Block
#'        None	None
#'        Stone	Stone
#' 	
#' MasVnrArea: Masonry veneer area in square feet
#' 
#' ExterQual: Evaluates the quality of the material on the exterior 
#' 		
#'        Ex	Excellent
#'        Gd	Good
#'        TA	Average/Typical
#'        Fa	Fair
#'        Po	Poor
#' 		
#' ExterCond: Evaluates the present condition of the material on the exterior
#' 		
#'        Ex	Excellent
#'        Gd	Good
#'        TA	Average/Typical
#'        Fa	Fair
#'        Po	Poor
#' 		
#' Foundation: Type of foundation
#' 		
#'        BrkTil	Brick & Tile
#'        CBlock	Cinder Block
#'        PConc	Poured Contrete	
#'        Slab	Slab
#'        Stone	Stone
#'        Wood	Wood
#' 		
#' BsmtQual: Evaluates the height of the basement
#' 
#'        Ex	Excellent (100+ inches)	
#'        Gd	Good (90-99 inches)
#'        TA	Typical (80-89 inches)
#'        Fa	Fair (70-79 inches)
#'        Po	Poor (<70 inches
#'        NA	No Basement
#' 		
#' BsmtCond: Evaluates the general condition of the basement
#' 
#'        Ex	Excellent
#'        Gd	Good
#'        TA	Typical - slight dampness allowed
#'        Fa	Fair - dampness or some cracking or settling
#'        Po	Poor - Severe cracking, settling, or wetness
#'        NA	No Basement
#' 	
#' BsmtExposure: Refers to walkout or garden level walls
#' 
#'        Gd	Good Exposure
#'        Av	Average Exposure (split levels or foyers typically score average or above)	
#'        Mn	Mimimum Exposure
#'        No	No Exposure
#'        NA	No Basement
#' 	
#' BsmtFinType1: Rating of basement finished area
#' 
#'        GLQ	Good Living Quarters
#'        ALQ	Average Living Quarters
#'        BLQ	Below Average Living Quarters	
#'        Rec	Average Rec Room
#'        LwQ	Low Quality
#'        Unf	Unfinshed
#'        NA	No Basement
#' 		
#' BsmtFinSF1: Type 1 finished square feet
#' 
#' BsmtFinType2: Rating of basement finished area (if multiple types)
#' 
#'        GLQ	Good Living Quarters
#'        ALQ	Average Living Quarters
#'        BLQ	Below Average Living Quarters	
#'        Rec	Average Rec Room
#'        LwQ	Low Quality
#'        Unf	Unfinshed
#'        NA	No Basement
#' 
#' BsmtFinSF2: Type 2 finished square feet
#' 
#' BsmtUnfSF: Unfinished square feet of basement area
#' 
#' TotalBsmtSF: Total square feet of basement area
#' 
#' Heating: Type of heating
#' 		
#'        Floor	Floor Furnace
#'        GasA	Gas forced warm air furnace
#'        GasW	Gas hot water or steam heat
#'        Grav	Gravity furnace	
#'        OthW	Hot water or steam heat other than gas
#'        Wall	Wall furnace
#' 		
#' HeatingQC: Heating quality and condition
#' 
#'        Ex	Excellent
#'        Gd	Good
#'        TA	Average/Typical
#'        Fa	Fair
#'        Po	Poor
#' 		
#' CentralAir: Central air conditioning
#' 
#'        N	No
#'        Y	Yes
#' 		
#' Electrical: Electrical system
#' 
#'        SBrkr	Standard Circuit Breakers & Romex
#'        FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
#'        FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
#'        FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
#'        Mix	Mixed
#' 		
#' 1stFlrSF: First Floor square feet
#'  
#' 2ndFlrSF: Second floor square feet
#' 
#' LowQualFinSF: Low quality finished square feet (all floors)
#' 
#' GrLivArea: Above grade (ground) living area square feet
#' 
#' BsmtFullBath: Basement full bathrooms
#' 
#' BsmtHalfBath: Basement half bathrooms
#' 
#' FullBath: Full bathrooms above grade
#' 
#' HalfBath: Half baths above grade
#' 
#' Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
#' 
#' Kitchen: Kitchens above grade
#' 
#' KitchenQual: Kitchen quality
#' 
#'        Ex	Excellent
#'        Gd	Good
#'        TA	Typical/Average
#'        Fa	Fair
#'        Po	Poor
#'        	
#' TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
#' 
#' Functional: Home functionality (Assume typical unless deductions are warranted)
#' 
#'        Typ	Typical Functionality
#'        Min1	Minor Deductions 1
#'        Min2	Minor Deductions 2
#'        Mod	Moderate Deductions
#'        Maj1	Major Deductions 1
#'        Maj2	Major Deductions 2
#'        Sev	Severely Damaged
#'        Sal	Salvage only
#' 		
#' Fireplaces: Number of fireplaces
#' 
#' FireplaceQu: Fireplace quality
#' 
#'        Ex	Excellent - Exceptional Masonry Fireplace
#'        Gd	Good - Masonry Fireplace in main level
#'        TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
#'        Fa	Fair - Prefabricated Fireplace in basement
#'        Po	Poor - Ben Franklin Stove
#'        NA	No Fireplace
#' 		
#' GarageType: Garage location
#' 		
#'        2Types	More than one type of garage
#'        Attchd	Attached to home
#'        Basment	Basement Garage
#'        BuiltIn	Built-In (Garage part of house - typically has room above garage)
#'        CarPort	Car Port
#'        Detchd	Detached from home
#'        NA	No Garage
#' 		
#' GarageYrBlt: Year garage was built
#' 		
#' GarageFinish: Interior finish of the garage
#' 
#'        Fin	Finished
#'        RFn	Rough Finished	
#'        Unf	Unfinished
#'        NA	No Garage
#' 		
#' GarageCars: Size of garage in car capacity
#' 
#' GarageArea: Size of garage in square feet
#' 
#' GarageQual: Garage quality
#' 
#'        Ex	Excellent
#'        Gd	Good
#'        TA	Typical/Average
#'        Fa	Fair
#'        Po	Poor
#'        NA	No Garage
#' 		
#' GarageCond: Garage condition
#' 
#'        Ex	Excellent
#'        Gd	Good
#'        TA	Typical/Average
#'        Fa	Fair
#'        Po	Poor
#'        NA	No Garage
#' 		
#' PavedDrive: Paved driveway
#' 
#'        Y	Paved 
#'        P	Partial Pavement
#'        N	Dirt/Gravel
#' 		
#' WoodDeckSF: Wood deck area in square feet
#' 
#' OpenPorchSF: Open porch area in square feet
#' 
#' EnclosedPorch: Enclosed porch area in square feet
#' 
#' 3SsnPorch: Three season porch area in square feet
#' 
#' ScreenPorch: Screen porch area in square feet
#' 
#' PoolArea: Pool area in square feet
#' 
#' PoolQC: Pool quality
#' 		
#'        Ex	Excellent
#'        Gd	Good
#'        TA	Average/Typical
#'        Fa	Fair
#'        NA	No Pool
#' 		
#' Fence: Fence quality
#' 		
#'        GdPrv	Good Privacy
#'        MnPrv	Minimum Privacy
#'        GdWo	Good Wood
#'        MnWw	Minimum Wood/Wire
#'        NA	No Fence
#' 	
#' MiscFeature: Miscellaneous feature not covered in other categories
#' 		
#'        Elev	Elevator
#'        Gar2	2nd Garage (if not described in garage section)
#'        Othr	Other
#'        Shed	Shed (over 100 SF)
#'        TenC	Tennis Court
#'        NA	None
#' 		
#' MiscVal: $Value of miscellaneous feature
#' 
#' MoSold: Month Sold (MM)
#' 
#' YrSold: Year Sold (YYYY)
#' 
#' SaleType: Type of sale
#' 		
#'        WD 	Warranty Deed - Conventional
#'        CWD	Warranty Deed - Cash
#'        VWD	Warranty Deed - VA Loan
#'        New	Home just constructed and sold
#'        COD	Court Officer Deed/Estate
#'        Con	Contract 15% Down payment regular terms
#'        ConLw	Contract Low Down payment and low interest
#'        ConLI	Contract Low Interest
#'        ConLD	Contract Low Down
#'        Oth	Other
#' 		
#' SaleCondition: Condition of sale
#' 
#'        Normal	Normal Sale
#'        Abnorml	Abnormal Sale -  trade, foreclosure, short sale
#'        AdjLand	Adjoining Land Purchase
#'        Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
#'        Family	Sale between family members
#'        Partial	Home was not completed when last assessed (associated with New Homes)
#' 
#' SalePrice: the property's sale price in dollars. This is the target variable that you're trying to predict.
#' 
#' ## 4. Methods and Analysis: Data exploration and visualization 
#' 
#' As you can see from the chart below, the sale prices in the test data set are right skewed. This is not unusual since the most expensive homes greater than 400K have lower volume of sales as compared with the bulk of sales in the 100K to 300K range. 
#' 
## ----message=FALSE, warning=FALSE, paged.print=FALSE---------------------

ggplot(data=train, aes(x=SalePrice)) +
  geom_histogram(fill="dark green", binwidth = 10000) +
  scale_x_continuous(breaks= seq(0, 800000, by=100000)) + 
  labs(title="House Sale Prices",
       caption = "source data: House Price train data")
        

#' 
#' In evaluating the dependent variables that are most important in predicting SalePrice I created a correlation matrix with SalePrice. The correlation matrix table below shows that there are 10 variables out of 37 numeric variables in the train dataset with a correlation of at least 0.5 and are greater than 0. 
#' 
## ----message=FALSE, warning=FALSE----------------------------------------
num_vars <- which(sapply(train, is.numeric)) #index vector numeric variables
num_vars_colnames <- data.table(names(num_vars)) #column names of the numeric variables

train_num_vars <- train[, num_vars]
cor_num_vars <- cor(train_num_vars, use="pairwise.complete.obs") #correlations of all numeric variables

#sort on decreasing correlations with SalePrice
cor_sorted <- as.matrix(sort(cor_num_vars[,'SalePrice'], decreasing = TRUE))
 #select only high corelations
high_cor <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0.5)))
high_cor_colnames <- data.table(high_cor)
  
cor_num_vars <- cor_num_vars[high_cor, high_cor]

corrplot(cor_num_vars, type = "upper")


#' 
#' As shown from the above dependent variable correlation matrix to SalePrice, OverallQual has the highest correlation at of the numeric variables at 0.8. We will plot this correlation in a box plot below to visualize the accuracy if OverallQual as the primary predictor of SalePrice. 
#' 
## ----message=FALSE, warning=FALSE----------------------------------------
ggplot(data=train[!is.na(train$SalePrice),], aes(x=factor(OverallQual), y=SalePrice))+
        geom_boxplot(col='blue') + labs(x='OverallQual') +
        scale_y_continuous(breaks= seq(0, 800000, by=100000))+
        labs(title="OverallQual correlation to SalePrice", caption = "source data: House Price train data")

#' 
#' 
#' 
#' ###   5. Evaluated Machine Learning Algorithms
#' ####      5.1 Model Prepping
#' 
#' Before we can start modeling, we need to first partition the SalesPrice dataset and here I will use the caret package.
#' 
## ------------------------------------------------------------------------
set.seed(123)

outcome <- train$SalePrice

partition <- createDataPartition(y=outcome,
                                 p=.5,
                                 list=F)
training <- train[partition,]
testing <- train[-partition,]


#' 
#' ####        5.2 Simple Linear Regression Model
#' 
#' 
#' The simple linear regression model that I will generate first uses the OverallQual bias or the *OverallQual effect* $b_o$ dependent variable on the training_set to predict the `SalePrice` $(Y)$. 
#' 
#' The *OverallQual effect model* is calculated as follows:
#' $$Y = \mu + b_o + \varepsilon$$
#' where:
#' 
#' #### * $(\mu)$  is the mean SalePrice for all Houses.   
#' #### * $(b_o)$  effects or bias, OverallQual effect. 
#' #### * $(\varepsilon)$ are independent errors sampled from the same distribution centered at 0.
#' 
#' The resulting RMSLE from this *OverallQual effect model* was 0.2690968. We can likely do better with adding more dependent variables to the model. 
#' 
## ----message=FALSE, warning=FALSE----------------------------------------
# Fitting Simple Regression Model to the train set.
set.seed(123)

OQ_effect_model <- lm(SalePrice ~ OverallQual, data = training)

summary(OQ_effect_model)

prediction <- predict(OQ_effect_model, testing, type="response")

prediction_log <- log(prediction)

testing_log <- log(testing$SalePrice)

model_rmse <- RMSE(testing_log, prediction_log)

RMSLE_table <- data_frame(Method = "Regression model using OverallQual effect", 
                               RMSLE = model_rmse)

RMSLE_table %>% knitr::kable(caption = "RMSLEs")


#' 
#' ####        5.3 Multiple Linear Regression Model
#' 
#' The next regression model that I will generate uses the the top 10 most correlated numeric variables to SalePrice on the train set to predict the `rating` $Y$ for the test set.
#' 
#' The *top-10-effect model* is calculated as follows:
#' $$Y_{u,i} = \mu + b_1 + b_2 + b_3 + b_4 + b_5 + b_6 + b_7 + b_8 + b_9 + b_{10} + \varepsilon$$
#' where:
#' 
#' #### * $(\mu)$  is the mean SalePrice for all Houses.   
#' #### * $(b_1)$  effects or bias, OverallQual effect. 
#' #### * $(b_2)$  effects or bias, GrLivArea effect. 
#' #### * $(b_3)$  effects or bias, GarageCars effect. 
#' #### * $(b_4)$  effects or bias, GarageArea effect. 
#' #### * $(b_5)$  effects or bias, TotalBsmtSF effect. 
#' #### * $(b_6)$  effects or bias, X1stFlrSF effect. 
#' #### * $(b_7)$  effects or bias, FullBath effect. 
#' #### * $(b_8)$  effects or bias, TotRmsAbvGrd effect. 
#' #### * $(b_9)$  effects or bias, YearBuilt effect. 
#' #### * $(b_10)$  effects or bias, YearRemodAdd effect. 
#' #### * $(\varepsilon)$ are independent errors sampled from the same distribution centered at 0. 
#' 
#' The resulting RMSLE from this *top-10-effect model* on the test_set was an improvement on the *OverallQual effect model* RMSLE at 0.1912581. Let's see if we can do better with regularization.
#' 
## ----message=FALSE, warning=FALSE----------------------------------------

set.seed(123)

top10_effect_model <- lm(SalePrice ~ OverallQual + GrLivArea + GarageCars +
                         GarageArea + TotalBsmtSF + X1stFlrSF + FullBath +
                         TotRmsAbvGrd + YearBuilt + YearRemodAdd, data = training)

summary(top10_effect_model)

prediction <- predict(top10_effect_model, testing, type="response")

prediction_log <- log(prediction)

testing_log <- log(testing$SalePrice)

model_rmse <- RMSE(testing_log, prediction_log)

RMSLE_table <- rbind(RMSLE_table,
                    data_frame(Method = "Regression model using top-10-effect", 
                               RMSLE = model_rmse))

RMSLE_table %>% knitr::kable(caption = "RMSLEs")

#' ####        5.4 Backward Elimination Linear Regression Model
#' 
#' The next regression model that I will generate uses backwards elimination to pick the most signification bias variables to SalePrice on the train set to predict the `rating` $Y$ for the test set.
#' 
#' The *backward-elimination model* is calculated by taking the 5 most significant variables from the top-10 model:
#' $$Y_{u,i} = \mu + b_1 + b_2 + b_3 + b_4 + b_5 + \varepsilon$$
#' where:
#' 
#' #### * $(\mu)$  is the mean SalePrice for all Houses.   
#' #### * $(b_1)$  effects or bias, OverallQual effect. 
#' #### * $(b_2)$  effects or bias, GrLivArea effect. 
#' #### * $(b_3)$  effects or bias, GarageCars effect. . 
#' #### * $(b_4)$  effects or bias, YearBuilt effect. 
#' #### * $(b_5)$  effects or bias, YearRemodAdd effect. 
#' #### * $(\varepsilon)$ are independent errors sampled from the same distribution centered at 0. 
#' 
#' The resulting RMSLE from this *backward elimination model* on the test_set was not an improvement and in a sense we did go backward at 0.2039798. Let's see if we can do better with regularization.
#' 
## ------------------------------------------------------------------------
set.seed(123)

top10_effect_model <- lm(SalePrice ~ OverallQual + GrLivArea + GarageCars + YearBuilt + YearRemodAdd, data = training)

summary(top10_effect_model)

prediction <- predict(top10_effect_model, testing, type="response")

prediction_log <- log(prediction)

testing_log <- log(testing$SalePrice)

model_rmse <- RMSE(testing_log, prediction_log)

RMSLE_table <- rbind(RMSLE_table,
                    data_frame(Method = "Regression model using backward elimination", 
                               RMSLE = model_rmse))

RMSLE_table %>% knitr::kable(caption = "RMSLEs")

#' 
#' ####        5.5 Random Forest Regression Model 
#' 
#' The next regression model that I will generate uses all variables to SalePrice on the training set using a randomForest algorithm.
#' 
#' The resulting RMSLE from this *Random Forest model* to predict SalePrice on the testing set resulted in a RMSLE of 0.1384136.
#' 
## ------------------------------------------------------------------------
set.seed(123)
rf_model <- randomForest(SalePrice ~ ., data = training)

prediction <- predict(rf_model, testing)

prediction_log <- log(prediction)

testing_log <- log(testing$SalePrice)

model_rmse <- RMSE(testing_log, prediction_log)

RMSLE_table <- rbind(RMSLE_table,
                    data_frame(Method = "Random Forest regression model",
                               RMSLE = model_rmse))

RMSLE_table %>% knitr::kable(caption = "RMSLEs")

#' 
#' ####        5.6 Lasso Regression Model
#' 
#' The Lasso Regression model on all variables in the training set to seeks to minimize the RMSLE using cross validation to pick the optimal λ.  
#' 
#' According to 'Statistics How To', in the Lasso Regression model, or Least Absolute Shrinkage and Selection Operator, it performs L1 regularization, which adds a penalty equal to the absolute value of the magnitude of coefficients. This type of regularization can result in sparse models with few coefficients; Some coefficients can become zero and eliminated from the model. A tuning parameter, λ controls the strength of the L1 penalty. λ is basically the amount of shrinkage.
#' 
## ----message=FALSE, warning=FALSE, paged.print=FALSE---------------------
# Lasso regularization method on the training set.

set.seed(123)

my_control <-trainControl(method="cv", number=5)
lassoGrid <- expand.grid(alpha = 1, lambda = seq(0.001,0.1,by = 0.0005))

lasso_model <- train(SalePrice ~ ., data = training, method='glmnet', trControl= my_control, tuneGrid=lassoGrid) 

lambda_opt <- lasso_model$bestTune

lassoVarImp <- varImp(lasso_model,scale=F)
lassoImportance <- lassoVarImp$importance

varsSelected <- length(which(lassoImportance$Overall!=0))
varsNotSelected <- length(which(lassoImportance$Overall==0))

cat('Lasso uses', varsSelected, 'variables in its model, and did not select', varsNotSelected, 'variables.')

prediction <- predict(lasso_model, training)

prediction_log <- log(prediction)

training_log <- log(training$SalePrice)

model_rmse <- RMSE(training_log, prediction_log)

RMSLE_table <- rbind(RMSLE_table,
                    data_frame(Method = "Lasso regression model", 
                               RMSLE = model_rmse))

RMSLE_table %>% knitr::kable(caption = "RMSLEs")


#' 
#' 
#' 
#' ## 6. Results:
#' 
#' The resulting RMSLE from this *Lasso regression model* on the training set brought the RMSLE down to 0.0978573. That is good enough for top 10 on the current Kaggle leaderboard! 
#' 
#' https://www.kaggle.com/c/house-prices-advanced-regression-techniques/leaderboard
#' 
#' 
#' ## 7. Conclusion:
#' 
#' The main learning objective of the HarvardX: Introduction to Data Science program was to give
#' aspiring data scientists like myself the tools in R to run analytic models using machine learning to make predictions and solve real world problems. This was a fascinating journey over the past six months and I look forward to improving my data science skills in R and Python through my work and personally through Kaggle competitions that I have an interest in. 
#' 
#' My "Harvard_Data_Science_House Prices Github repository" is **[in this link](https://github.com/jnielsonresearch/Harvard_Data_Science_Capstone_HousePrices)**
#' 
#' ## *References*
#' 
#' #### * Irizzary,R., 2019. Introduction to Data Science,
#' ####   github page,https://rafalab.github.io/dsbook/
#' #### * Statistics How To, https://www.statisticshowto.datasciencecentral.com/lasso-regression/
#' 
#' 
