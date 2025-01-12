Create a clean_data function and use it as input for the price and returns calculations.

def clean_data():
    try:
        input_file_path = filedialog.askopenfilename(title="Select Input CSV File")
        output_file_path = filedialog.asksaveasfilename(title="Save Cleaned CSV File As", defaultextension=".csv")

        if not input_file_path or not output_file_path:
            messagebox.showwarning("File not selected", "Please select valid input and output files.")
            return

        # Read the CSV into a DataFrame
        df = pd.read_csv(input_file_path)

        # Get user input for tickers
        input_tickers = tickers_entry.get()
        if not input_tickers:
            messagebox.showwarning("Input Required", "Please enter ticker symbols.")
            return

        ticker_list = [ticker.strip() for ticker in input_tickers.split(",")]

        # Filter the data based on tickers
        df_filtered = df[df["<Ticker>"].isin(ticker_list)]

        # Remove unwanted columns
        df_filtered = df_filtered.drop(columns=["<Open>", "<High>", "<Low>", "<Volume>"])

        # Rename the <DTYYYYMMDD> column to Date
        df_filtered.rename(columns={"<DTYYYYMMDD>": "Date"}, inplace=True)

        # Pivot the data to have tickers as columns and their respective Close prices
        df_pivot = df_filtered.pivot(index="Date", columns="<Ticker>", values="<Close>")

        # Save the cleaned data to a new CSV file
        df_pivot.to_csv(output_file_path)
        messagebox.showinfo("Success", f"Cleaned data saved to {output_file_path}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")