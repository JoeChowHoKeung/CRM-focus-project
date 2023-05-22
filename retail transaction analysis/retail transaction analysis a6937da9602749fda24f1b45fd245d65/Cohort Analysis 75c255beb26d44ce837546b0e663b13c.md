# Cohort Analysis

This document outlines a cohort analysis project for evaluating customer retention and purchasing behavior. The project's objective is to develop a mechanism to assess the effectiveness of the customer retention system, identify customer purchasing patterns in different lifetime stages, and examine trends throughout the customer's lifetime. The methodology for this project includes cohort assignment based on the customer's first acquisition date, identifying active customers, and examining retention ability and consumption behavior. The findings of this project include life-time consumption, sales trends, and customer retention. The data processing for this project includes cohort creation, cohort aggregation, and cohort analysis for life-time consumption, sales trends, and retention rate. The report concludes with a summary of the project in 150 words and a recommendation to present the results to the supervisor. The project aims to provide insights into customer behavior and retention for future marketing and sales strategy development.

## Objective

- To produce a mechanism to evaluate the performance of a customer retention system.
- To discover the customer purchasing behavior in different customer lifetime stages.
- To examine the common or special trend throughout the life time.

# Methodology

### **Cohort Assignment**

To achieve this objective, we will first define our cohorts based on the **customer first acquisition date**. Those new customer would be assigned into a monthly-based cohort for tracking their consumption activity. 

### Active Customers Definition

Once the customer made a purchase at the current month, customer would be identified as an active customer; otherwise, it would be a inactive customer.

### Examination

- **Retention Ability**
    
    The cohort analysis model is effective to track whether the customer has a continuous purchasing behavior.  In accordance with the changes during examination, we could have an insight about
    
- **Consumption Behavior**
    
    The continuous observation with a same group of customers provides a easy way to understand the common consumption pattern. Alternatively, cross-comparing with cohort, we could find out the outstanding groups of customer for taking in-depth understanding about them.  
    

## Findings

- **Life-Time Consumption**
    
    ![496087e2-5546-46e0-b9c8-b5763d2ea8a3.png](Cohort%20Analysis%2075c255beb26d44ce837546b0e663b13c/496087e2-5546-46e0-b9c8-b5763d2ea8a3.png)
    
    - **Code**
        
        
        ```python
        # plot the cohort analysis by cumulative sales, percentage change of cumulative sales
        
        import seaborn as sns
        import matplotlib.pyplot as plt
        # get the pivot_table in forms of cohort, invoicedate 
        # to present the summation of total_sales with percentage change along the column
        vis_df = cohort_df.pivot_table(
            values='total_sales',
            index='cohort',
            columns='invoicedate',
            aggfunc='sum'
        ).cumsum(axis=1).pct_change(axis=1).loc[:, '2010-12':'2011-11']
        
        # plot the heatmap
        plt.figure(figsize=(10, 5))
        sns.heatmap(
            vis_df,
            annot=True,
            fmt='.0%',
            cmap='RdYlGn_r',
            vmin=-1,
            vmax=1
        )
        plt.xlabel('Transaction Month')
        plt.ylabel('Cohort')
        plt.title('Cohort Analysis by Cumulative Sales, Percentage Change of Cumulative Sales')
        plt.show()
        ```
        
    - In general, each cohort experiences stable growth in the range of 10% to 20% every month.
    - However, one cohort of August 2011 in particular, with outstanding performance, experienced a growth rate between 33% and 40%, which is double that of the general performance.
    - Additionally, in May 2011, there was a significant increase from around 16% to over 20%. This presents an interesting opportunity for in-depth study.
- **Sales Trends**
    
    ![0f399f31-aafb-4e64-8aa5-af8a9451f735.png](Cohort%20Analysis%2075c255beb26d44ce837546b0e663b13c/0f399f31-aafb-4e64-8aa5-af8a9451f735.png)
    
    - **Code**
        
        
        ```python
        # plot the cohort analysis by total of sales, percentage change of sales
        
        # get the pivot_table in forms of cohort, invoicedate
        # to present the summation of total_sales with percentage change along the column
        vis_df = cohort_df.pivot_table(
            values='total_sales',
            index='cohort',
            columns='invoicedate',
            aggfunc='sum'
        ).pct_change(axis=1).loc[:, '2010-12':'2011-11']
        
        # plot the heatmap
        plt.figure(figsize=(10, 5))
        sns.heatmap(
            vis_df,
            annot=True,
            fmt='.0%',
            cmap='RdYlGn_r',
            vmin=-1,
            vmax=1
        )
        plt.xlabel('Transaction Month')
        plt.ylabel('Cohort')
        plt.title('Cohort Analysis by Total of Sales, Percentage Change of Sales')
        plt.show()
        ```
        
    - In general, the second month of each cohort may experience a significant decrease in sales. However, typically, the two to five months after the first purchase see a rise of a certain percentage. This results in a long period of consumption cycle.
    - Along with the transaction month, there are two identifiable red columns in May 2011 and September 2011 which indicate over a 50% increase in sales compared to the previous transaction month. After the sales peak, the overall cohort would calm down for a certain period. We believe that the general consumption behavior may be affected by specific events or quarterly cyclical factors.
    - Some cohorts, *such as cohort(2010-12) and cohort(2011-02)*,  exhibit stable and positive sales growth. There may be a slight decrease excepting for the second month from the start, but overall there is a significant boost to the growth.
- **Customer Retention**
    
    ![16d6c85c-4295-4431-ad06-b44d4956eb24.png](Cohort%20Analysis%2075c255beb26d44ce837546b0e663b13c/16d6c85c-4295-4431-ad06-b44d4956eb24.png)
    
    - **Code**
        
        
        ```python
        # plot the cohort analysis by customer retention rate
        
        # This code creates a pivot table that shows the number of customers in each cohort for each month. 
        # Then, it divides each cell by the number of customers in the first month of the cohort to get the retention rate. 
        #Finally, it uses seaborn to create a heatmap that shows the retention rate for each cohort for each month.
        
        vis_df = cohort_df.pivot_table(
            values='n_customers',
            index='cohort',
            columns='invoicedate',
            aggfunc='sum'
        ).divide(
            cohort_df.reset_index(level=1).groupby('cohort')['n_customers'].transform('first').drop_duplicates(),
            axis=0
        ).loc[:, '2010-12':'2011-11']
        
        plt.figure(figsize=(10, 5))
        sns.heatmap(
            vis_df,
            annot=True,
            fmt='.0%',
            cmap='RdYlGn_r',
            vmin=0,
            vmax=1
        )
        plt.title('Cohort Analysis by Customer Retention Rate')
        plt.xlabel('Transaction Month')
        plt.ylabel('Cohort')
        plt.show()
        ```
        
    - In general, each cohort has 20% of customers active in a given transaction month. However, the cohort from December 2010 has a strong retention rate, with 33-49% of customers remaining active throughout the year.
    - In the transaction month of 2011-08, there is a noticeable deep green area, indicating that fewer transactions occurred.

# Data Processing

1. **Cohort Creation**
    
    For each customer, the first purchase within the timeframe would be treated as new customers. The date is formatted as `yyyy-mm` . The new customer would be assigned into a cohort which is a group of new customer made a first purchase within the same transaction month. 
    
    ```python
    # get the first purchase date of each customer
    cohort = (
        df
        # group by customerid and get the first purchase date
        .groupby('customerid')['invoicedate']
        # get the first purchase date
        .min()
        # get the year and month of the first purchase date
        .transform(lambda x: x.strftime('%Y-%m'))
        # rename the column
        .rename('cohort')
        # convert the column to string
        .astype('str')
    )
    ```
    
2. **Cohort Aggregation**
    
    ```python
    cohort_df = (
        df
        # join the cohort to the dataset
        .join(cohort, on='customerid')
        .assign(
            # get the year and month of each transaction
            invoicedate = df['invoicedate'].dt.strftime('%Y-%m'),
            # get the total of transaction amount of each transaction
            sales = df['quantity'] * df['unitprice']
            )
        # group by cohort and invoicedate
        .groupby(['cohort', 'invoicedate'])
        .agg(
            # get the number of unique customerid
            n_customers=('customerid', 'nunique'),
            # get the total of transaction amount
            total_sales=('sales', 'sum'),
            # get the average of transaction amount
            average_sales=('sales', 'mean')
            )
    )
    ```
    
3. **Cohort Analysis for Life-Time Consumption**
4. **Cohort Analysis for Sales Trends**
5. **Cohort Analysis for Retention Rate**