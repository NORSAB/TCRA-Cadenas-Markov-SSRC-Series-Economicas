import matplotlib.dates as mdates

def style_subplot(ax, title, date_min, date_max, y_limit_max=170):
    """
    Applies consistent styling to a matplotlib subplot.
    Aplica un estilo consistente a un subplot de matplotlib.

    Args:
        ax (matplotlib.axes.Axes): The subplot to style.
        title (str): The title for the subplot.
        date_min (pd.Timestamp): The minimum date for the x-axis limit and vline.
        date_max (pd.Timestamp): The maximum date for the x-axis limit and vline.
        y_limit_max (int, optional): The upper limit for the y-axis. Defaults to 170.
    """
    # Titulos y etiquetas
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_ylabel("Lempiras (HNL)", fontsize=10)

    # Limites de ejes
    ax.set_ylim(0, y_limit_max)
    ax.set_xlim(date_min, date_max)

    # Add vertical reference lines for start and end dates
    ax.axvline(x=date_min, color='gray', linestyle='--', linewidth=2)
    ax.axvline(x=date_max, color='gray', linestyle='--', linewidth=2)

    # Add date annotations dynamically positioned near the top of the plot
    y_pos_text = y_limit_max * 0.97  # Position text at 97% of the y-axis height
    ax.text(date_min, y_pos_text, date_min.strftime('%m-%d-%Y'), fontsize=10, ha='left', va='top')
    ax.text(date_max, y_pos_text, date_max.strftime('%m-%d-%Y'), fontsize=10, ha='right', va='top')

    # Format x-axis to show years only
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # Configure tick parameters
    ax.tick_params(axis='x', labelrotation=0, labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
