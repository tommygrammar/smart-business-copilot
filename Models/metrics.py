from Data.business_data import historical_data

def how_much_revenue(period):
    revenue = historical_data['revenue'][-period] * 100
    narrate = f"ulimake KES {revenue:.1f} "
    return narrate

def how_much_sales(period):
    sales = historical_data['sales'][-period]
    narrate = f"uliuza units {sales:.1f}"
    return narrate


def how_much_inventory(period):
    inventory = historical_data['inventory'][-period]
    narrate = f"sai uko na inventory ya {inventory:.1f} units"
    return narrate