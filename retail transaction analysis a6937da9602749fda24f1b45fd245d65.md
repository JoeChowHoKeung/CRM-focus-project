# retail transaction analysis

Under: Side Porject

# Data Description

[OnlineRetail.csv](retail%20transaction%20analysis%20a6937da9602749fda24f1b45fd245d65/OnlineRetail.csv)

[Reference Link](https://www.kaggle.com/code/kirshoff/rfm-cohort-analysis-project)

- Rows: 541909
- Columns: 8
- Data Time Frame : *01-Dec-2010* to *09-Dec-2011*
- The Observation Country : United Kingdom

| Column Name | Data Type | Description |
| --- | --- | --- |
| invoiceno | string | The identification code of transactions |
| stockcode | string | The SKU number of the product in the transaction |
| description | string | The text description of the product |
| quantity | integer | The quantity sold in the transaction |
| invoicedate | datetime | The date of the transaction |
| unitprice | float | The average price of the product |
| customerid | string | The identification of the customer who made the transaction |
| country | string | The name of the country in which the transaction took place |

### Data Cleansing and Selection

```python
data_dir = 'OnlineRetail.csv'
import pandas as pd

# loading data to pandas dataframe
raw = (
		pd
		.read_csv(data_dir, engine='pyarrow')
		# rename columns into lower case
		.rename(mapper=lambda col: col.lower(), axis='columns')
		)

# reformating the data columns into proper data types
raw = (
    raw
    .assign(
    # convert the date to datetime format
    invoicedate = pd.to_datetime(raw['invoicedate'], format='%d-%m-%Y %H:%M'),
    # convert the description from btyes to string
    description = raw['description'].str.decode('unicode_escape'),
    # convert customerID to string
    customerid = raw['customerid'].astype("string").str.replace('.0', '')
    )
)

# only observe the transaction within the region of United Kingdom
df = raw.query('country in "United Kingdom"')
```

| invoiceno | stockcode | description | quantity | invoicedate | unitprice | customerid | country |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 536365 | 85123A | WHITE HANGING HEART T-LIGHT HOLDER | 6 | @December 1, 2010 08:26:00 | 2.55 | 17850 | United Kingdom |
| 536365 | 71053 | WHITE METAL LANTERN | 6 | @December 1, 2010 08:26:00 | 3.39 | 17850 | United Kingdom |
| 536365 | 84406B | CREAM CUPID HEARTS COAT HANGER | 8 | @December 1, 2010 08:26:00 | 2.75 | 17850 | United Kingdom |
| 536365 | 84029G | KNITTED UNION FLAG HOT WATER BOTTLE | 6 | @December 1, 2010 08:26:00 | 3.39 | 17850 | United Kingdom |
| 536365 | 84029E | RED WOOLLY HOTTIE WHITE HEART. | 6 | @December 1, 2010 08:26:00 | 3.39 | 17850 | United Kingdom |

<aside>
👉 About the data selection, based on the reasons of the completeness of transaction data within the timeframe and domestic factor, we select the UK as the observation country for our project.

</aside>

# Business Problem

- **Customer Segmentation**
    
    [RFM Clustering](retail%20transaction%20analysis%20a6937da9602749fda24f1b45fd245d65/RFM%20Clustering%20391e432790e84abfa775617e6ea7b54b.md)
    

- **Cohort Analysis**
    
    [Cohort Analysis](retail%20transaction%20analysis%20a6937da9602749fda24f1b45fd245d65/Cohort%20Analysis%2075c255beb26d44ce837546b0e663b13c.md)
    

- **Product Categories Clustering**