import matplotlib.pyplot as plt
import pandas as pd

from polynomial_fitting import PolynomialFitting


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    #filter data
    cd = pd.read_csv(filename, parse_dates=['Date'])
    cd['DayOfYear'] = cd['Date'].dt.dayofyear
    cd = cd[cd['Temp'] >= -20]
    cd.drop_duplicates(inplace=True)
    return cd



if __name__ == '__main__':

    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")
    israel_df = df[df['Country']=='Israel'].copy()
    years = sorted(israel_df['Year'].unique())
    cmap = plt.colormaps['tab20']

    #create graph
    plt.figure(figsize=(14, 8))
    for i, year in enumerate(years):
        year_data = israel_df[israel_df['Year'] == year]
        plt.scatter(year_data['DayOfYear'], year_data['Temp'], color=cmap(i), label=str(year), s=10)
    plt.title("Temperature in Israel by DayOfYear (Colored by Year)")
    plt.xlabel("DayOfYear")
    plt.ylabel("Temperature")
    plt.legend(title = "Year", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("Temperature_in_Israel_by_DayOfYear(Colored by Year).png")


    # Question 3 - Exploring data for specific country
    months = sorted(israel_df['Month'].unique())
    monthly_std = israel_df.groupby('Month')['Temp'].agg('std')
    #create graph
    plt.figure(figsize=(14, 8))
    plt.bar(monthly_std.index, monthly_std.values, color="salmon")
    plt.xticks(range(1,13))
    plt.xlabel("Month")
    plt.ylabel("Standard Deviation of Temperature")
    plt.title("Standard Deviation of Temperature by Month in Israel")
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
    plt.tight_layout()
    plt.savefig("Standard Deviation of Temperature by Month in Israel.png")

    # Question 4 - Exploring differences between countries
    countries = sorted(df['Country'].unique())
    grouped = df.groupby(['Country', 'Month'])['Temp'].agg(['mean', 'std']).reset_index()
    #create graph
    plt.figure(figsize=(14, 8))
    for i, country in enumerate(countries):
        country_data = grouped[grouped['Country'] == country]
        plt.errorbar(country_data['Month'], country_data['mean'], yerr=country_data['std'], label=country, color=cmap(i))
    plt.title("Average Temperature by Month in Different Countries")
    plt.xlabel("Month")
    plt.ylabel("Average Temperature")
    plt.xticks(range(1,13))
    plt.legend(title = "Country")
    plt.grid(axis = 'y', linestyle = '--', alpha = 0.7)
    plt.tight_layout()
    plt.savefig("Average Temperature by Month in Different Countries.png")

    # Question 5 - Fitting model for different values of `k`
    random_seed = 54
    train_data = israel_df.sample(frac = 0.75, random_state=random_seed)
    test_data = israel_df.drop(train_data.index)
    test_errors = []
    train_X = train_data['DayOfYear']
    train_y = train_data['Temp']
    for k in range(1,11):
        model = PolynomialFitting(k)
        model.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = model.loss(test_data['DayOfYear'].to_numpy(), test_data['Temp'].to_numpy())
        test_errors.append(loss)
        print(f"k={k}, loss={loss}")
    #create graph
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, 11), test_errors, color='salmon')
    plt.xlabel("Polynomial Degree (k)")
    plt.ylabel("Loss)")
    plt.title("Test Error for Different Polynomial Degrees")
    plt.xticks(range(1, 11))
    plt.tight_layout()
    plt.savefig("Test Error for Different Polynomial Degrees.png")

    # Question 6 - Evaluating fitted model on different countries
    best_model = PolynomialFitting(5)
    best_model.fit(train_X.to_numpy(), train_y.to_numpy())
    losses = {}
    countries = sorted(df['Country'].unique())
    for country in countries:
        if country == 'Israel':
            continue
        country_data = df[df['Country'] == country]
        loss = best_model.loss(country_data['DayOfYear'].to_numpy(), country_data['Temp'].to_numpy())
        losses[country] = loss
    #create graph
    plt.figure(figsize=(10, 6))
    plt.bar(losses.keys(), losses.values(), color='salmon')
    plt.title(f"Model Loss on Other Countries (Trained on Israel, Degree 5)")
    plt.xlabel("Country")
    plt.ylabel("Loss")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Model Loss on Other Countries (Trained on Israel, Degree 5).png")
