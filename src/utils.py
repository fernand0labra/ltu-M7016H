import matplotlib.pyplot as plt

def char_plot(sizes, labels):
    colors = ['yellowgreen', 'gold', 'lightskyblue']

    plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')

    # Add circle at center to create the effect of a cheese graph
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title('Dataset Class Division')
    plt.axis('equal')  # Ensure circular pie
    plt.tight_layout()  # Avoid overlapping labels
    plt.show()