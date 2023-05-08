import matplotlib.pyplot as plt

def char_plot(set, sizes, labels):
    colors = ['yellowgreen', 'gold', 'lightskyblue']

    plt.figure(figsize=(5, 4))
    plt.pie(sizes, labels=labels, colors=colors, startangle=90, autopct='%1.1f%%')

    # Add circle at center to create the effect of a cheese graph
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title(set.capitalize() + ' Set Class Division')
    plt.axis('equal')  # Ensure circular pie
    plt.tight_layout()  # Avoid overlapping labels
    plt.show()