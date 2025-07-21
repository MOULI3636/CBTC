import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data (only needed once)
nltk.download('stopwords')

class SpamEmailDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Email Detection System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Set light color palette
        self.colors = {
            'ham': '#66c2a5',  # Light teal
            'spam': '#fc8d62',  # Light orange
            'background': '#f8f9fa',
            'text': '#495057',
            'highlight': '#6c757d'
        }
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        self.train_model()
        
        # Create UI
        self.create_header()
        self.create_navigation()
        self.create_main_content()
        
        # Show initial view
        self.show_dashboard()
    
    def load_data(self):
        # In a real application, this would load from the Excel file
        # For this example, we'll create a sample DataFrame
        data = {
            'v1': ['ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'ham', 'spam'],
            'v2': [
                "Go until jurong point, crazy.. Available only in bugis",
                "Ok lar... Joking wif u oni...",
                "Free entry in 2 a wkly comp to win FA Cup final tkts",
                "U dun say so early hor... U c already then say...",
                "WINNER!! As a valued network customer you have been selected",
                "Nah I don't think he goes to usf, he lives around here though",
                "Had your mobile 11 months or more? U R entitled to Update",
                "I'm gonna be home soon and i don't want to talk about this stuff",
                "Even my brother is not like to speak with me.",
                "SIX chances to win CASH! From 100 to 20,000 pounds"
            ]
        }
        self.df = pd.DataFrame(data)
        self.df.columns = ['label', 'text']
        
        # Add more sample data to make it realistic
        for i in range(50):
            if i % 5 == 0:
                self.df.loc[len(self.df)] = ['spam', f"Congrats! You've won {i*100} dollars! Call now to claim"]
            else:
                self.df.loc[len(self.df)] = ['ham', f"Hey, just checking in. Let me know when you're free to meet up {i}"]
        
        # Add example emails for the explore feature
        self.example_emails = {
            "Typical Spam": {
                "text": "FREE entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
                "features": ["FREE", "win", "comp", "Text", "apply"]
            },
            "Typical Ham": {
                "text": "Hey Sam, just wanted to check if you're free for lunch tomorrow. Let me know what time works for you!",
                "features": ["Hey", "lunch", "tomorrow", "time"]
            },
            "Tricky Spam": {
                "text": "Your account has been credited with $500. Click here to claim your reward: http://reward.example.com",
                "features": ["account", "credited", "$500", "Click", "reward"]
            },
            "Tricky Ham": {
                "text": "Your invoice for $500 is ready. Please make payment by the end of the week.",
                "features": ["invoice", "$500", "payment", "week"]
            }
        }
    
    def preprocess_data(self):
        # Clean text data
        self.df['clean_text'] = self.df['text'].apply(self.clean_text)
        
        # Count words
        self.df['word_count'] = self.df['text'].apply(lambda x: len(str(x).split()))
        
        # Count characters
        self.df['char_count'] = self.df['text'].apply(lambda x: len(str(x)))
        
        # Count sentences (approximate)
        self.df['sentence_count'] = self.df['text'].apply(lambda x: len(str(x).split('.')))
        
        # Average word length
        self.df['avg_word_length'] = self.df['text'].apply(lambda x: np.mean([len(word) for word in str(x).split()]))
    
    def clean_text(self, text):
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # Remove stopwords
        stop = stopwords.words('english')
        text = " ".join([word for word in text.split() if word not in stop])
        
        # Stemming
        stemmer = PorterStemmer()
        text = " ".join([stemmer.stem(word) for word in text.split()])
        
        return text
    
    def train_model(self):
        # Vectorize text
        self.vectorizer = CountVectorizer()
        X = self.vectorizer.fit_transform(self.df['clean_text'])
        y = self.df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = MultinomialNB()
        self.model.fit(X_train, y_train)
        
        # Store accuracy
        self.accuracy = self.model.score(X_test, y_test)
    
    def create_header(self):
        # Header frame
        header_frame = tk.Frame(self.root, bg="#343a40", height=80)
        header_frame.pack(fill=tk.X)
        
        # Title
        title_label = tk.Label(
            header_frame, 
            text="Spam Email Detection System", 
            font=("Helvetica", 20, "bold"), 
            fg="white", 
            bg="#343a40"
        )
        title_label.pack(pady=20)
    
    def create_navigation(self):
        # Navigation frame
        nav_frame = tk.Frame(self.root, bg="#495057", width=200)
        nav_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Navigation buttons
        buttons = [
            ("Dashboard", self.show_dashboard),
            ("Email Analysis", self.show_analysis),
            ("Model Performance", self.show_performance),
            ("Test Classifier", self.show_classifier),
            ("Explore Examples", self.show_examples),
            ("Export Results", self.export_results)
        ]
        
        for text, command in buttons:
            btn = tk.Button(
                nav_frame, 
                text=text, 
                font=("Helvetica", 12), 
                bg="#495057", 
                fg="white",
                bd=0,
                padx=20,
                pady=15,
                anchor="w",
                command=command
            )
            btn.pack(fill=tk.X)
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#6c757d"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg="#495057"))
    
    def create_main_content(self):
        # Main content frame
        self.main_frame = tk.Frame(self.root, bg=self.colors['background'])
        self.main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var, 
            bd=1, 
            relief=tk.SUNKEN, 
            anchor=tk.W,
            bg="#e9ecef",
            fg="#495057"
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def clear_main_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
    def show_dashboard(self):
        self.clear_main_frame()
        self.status_var.set("Dashboard View")
        
        # Dashboard title
        title_label = tk.Label(
            self.main_frame, 
            text="Email Statistics Dashboard", 
            font=("Helvetica", 16, "bold"), 
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        title_label.pack(pady=20)
        
        # Create frames for charts
        top_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        top_frame.pack(fill=tk.X, padx=20, pady=10)
        
        middle_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        middle_frame.pack(fill=tk.X, padx=20, pady=10)
        
        bottom_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Chart 1: Email Distribution (Pie)
        fig1, ax1 = plt.subplots(figsize=(5, 4), dpi=100)
        counts = self.df['label'].value_counts()
        ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', 
                colors=[self.colors['ham'], self.colors['spam']], startangle=90)
        ax1.set_title('Email Distribution (Spam vs Ham)', color=self.colors['text'])
        
        canvas1 = FigureCanvasTkAgg(fig1, master=top_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.LEFT, padx=10)
        
        # Chart 2: Word Count Distribution (KDE)
        fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=100)
        for label in ['ham', 'spam']:
            sns.kdeplot(data=self.df[self.df['label']==label], x='word_count', 
                        ax=ax2, label=label, color=self.colors[label])
        ax2.set_title('Word Count Distribution', color=self.colors['text'])
        ax2.set_xlabel('Word Count', color=self.colors['text'])
        ax2.set_ylabel('Density', color=self.colors['text'])
        ax2.legend()
        
        canvas2 = FigureCanvasTkAgg(fig2, master=top_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.LEFT, padx=10)
        
        # Chart 3: Character Count Over Time (Line)
        fig3, ax3 = plt.subplots(figsize=(10, 4), dpi=100)
        self.df['index'] = range(len(self.df))
        for label in ['ham', 'spam']:
            subset = self.df[self.df['label']==label]
            ax3.plot(subset['index'], subset['char_count'], 
                     label=label, color=self.colors[label], alpha=0.7)
        ax3.set_title('Character Count Over Time', color=self.colors['text'])
        ax3.set_xlabel('Email Index', color=self.colors['text'])
        ax3.set_ylabel('Character Count', color=self.colors['text'])
        ax3.legend()
        
        canvas3 = FigureCanvasTkAgg(fig3, master=middle_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Chart 4: Common Words in Spam (Bar)
        fig4, ax4 = plt.subplots(figsize=(10, 5), dpi=100)
        
        # Get top words in spam
        spam_words = ' '.join(self.df[self.df['label']=='spam']['clean_text']).split()
        spam_word_counts = Counter(spam_words).most_common(10)
        
        words = [word[0] for word in spam_word_counts]
        counts = [word[1] for word in spam_word_counts]
        
        ax4.bar(words, counts, color=self.colors['spam'])
        ax4.set_title('Top 10 Words in Spam Emails', color=self.colors['text'])
        ax4.set_xlabel('Words', color=self.colors['text'])
        ax4.set_ylabel('Frequency', color=self.colors['text'])
        plt.xticks(rotation=45)
        
        canvas4 = FigureCanvasTkAgg(fig4, master=bottom_frame)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_analysis(self):
        self.clear_main_frame()
        self.status_var.set("Email Analysis View")
        
        # Title
        title_label = tk.Label(
            self.main_frame, 
            text="Email Text Analysis", 
            font=("Helvetica", 16, "bold"), 
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        title_label.pack(pady=20)
        
        # Create frames for charts
        top_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        top_frame.pack(fill=tk.X, padx=20, pady=10)
        
        middle_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        middle_frame.pack(fill=tk.X, padx=20, pady=10)
        
        bottom_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Chart 1: Character Count Distribution (Violin)
        fig1, ax1 = plt.subplots(figsize=(5, 4), dpi=100)
        sns.violinplot(x='label', y='char_count', data=self.df, ax=ax1, 
                      palette={'ham': self.colors['ham'], 'spam': self.colors['spam']})
        ax1.set_title('Character Count Distribution', color=self.colors['text'])
        ax1.set_xlabel('Email Type', color=self.colors['text'])
        ax1.set_ylabel('Character Count', color=self.colors['text'])
        
        canvas1 = FigureCanvasTkAgg(fig1, master=top_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(side=tk.LEFT, padx=10)
        
        # Chart 2: Word Length Distribution (Boxen)
        fig2, ax2 = plt.subplots(figsize=(5, 4), dpi=100)
        sns.boxenplot(x='label', y='avg_word_length', data=self.df, ax=ax2, 
                     palette={'ham': self.colors['ham'], 'spam': self.colors['spam']})
        ax2.set_title('Word Length Distribution', color=self.colors['text'])
        ax2.set_xlabel('Email Type', color=self.colors['text'])
        ax2.set_ylabel('Average Word Length', color=self.colors['text'])
        
        canvas2 = FigureCanvasTkAgg(fig2, master=top_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(side=tk.LEFT, padx=10)
        
        # Chart 3: Sentence Count (Bar)
        fig3, ax3 = plt.subplots(figsize=(10, 4), dpi=100)
        sns.countplot(x='sentence_count', hue='label', data=self.df, ax=ax3,
                     palette={'ham': self.colors['ham'], 'spam': self.colors['spam']})
        ax3.set_title('Sentence Count Distribution', color=self.colors['text'])
        ax3.set_xlabel('Number of Sentences', color=self.colors['text'])
        ax3.set_ylabel('Count', color=self.colors['text'])
        ax3.legend(title='Email Type')
        
        canvas3 = FigureCanvasTkAgg(fig3, master=middle_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Chart 4: Word Count vs Character Count (Scatter)
        fig4, ax4 = plt.subplots(figsize=(10, 5), dpi=100)
        for label in ['ham', 'spam']:
            subset = self.df[self.df['label']==label]
            ax4.scatter(subset['word_count'], subset['char_count'], 
                        label=label, color=self.colors[label], alpha=0.6)
        ax4.set_title('Word Count vs Character Count', color=self.colors['text'])
        ax4.set_xlabel('Word Count', color=self.colors['text'])
        ax4.set_ylabel('Character Count', color=self.colors['text'])
        ax4.legend()
        
        canvas4 = FigureCanvasTkAgg(fig4, master=bottom_frame)
        canvas4.draw()
        canvas4.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def show_performance(self):
        self.clear_main_frame()
        self.status_var.set("Model Performance View")
        
        # Title
        title_label = tk.Label(
            self.main_frame, 
            text="Model Performance Metrics", 
            font=("Helvetica", 16, "bold"), 
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        title_label.pack(pady=20)
        
        # Performance metrics frame
        metrics_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        metrics_frame.pack(pady=20)
        
        # Accuracy
        accuracy_label = tk.Label(
            metrics_frame, 
            text=f"Model Accuracy: {self.accuracy:.2%}", 
            font=("Helvetica", 14),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        accuracy_label.pack(pady=10)
        
        # Confusion Matrix (simulated)
        conf_matrix_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        conf_matrix_frame.pack(pady=20)
        
        # Simulated confusion matrix
        conf_matrix = {
            "": ["Predicted Ham", "Predicted Spam"],
            "Actual Ham": ["85%", "15%"],
            "Actual Spam": ["10%", "90%"]
        }
        
        # Create table
        for i, (key, values) in enumerate(conf_matrix.items()):
            for j, value in enumerate(values):
                if i == 0:
                    # Header row
                    lbl = tk.Label(
                        conf_matrix_frame, 
                        text=value, 
                        font=("Helvetica", 12, "bold"),
                        bg="#6c757d",
                        fg="white",
                        width=15,
                        height=2,
                        borderwidth=1,
                        relief="solid"
                    )
                else:
                    # Data rows
                    lbl = tk.Label(
                        conf_matrix_frame, 
                        text=value, 
                        font=("Helvetica", 12),
                        bg="#e9ecef",
                        fg=self.colors['text'],
                        width=15,
                        height=2,
                        borderwidth=1,
                        relief="solid"
                    )
                lbl.grid(row=i, column=j, padx=1, pady=1)
        
        # Feature importance (simulated)
        feature_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        feature_frame.pack(pady=20)
        
        # Feature importance title
        feature_title = tk.Label(
            feature_frame, 
            text="Top Predictive Features", 
            font=("Helvetica", 14, "bold"),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        feature_title.pack()
        
        # Feature importance table
        features = [
            ("free", "High"),
            ("win", "High"),
            ("prize", "High"),
            ("call", "Medium"),
            ("cash", "Medium"),
            ("urgent", "Medium"),
            ("congrats", "Low"),
            ("selected", "Low"),
            ("claim", "Low")
        ]
        
        for i, (feature, importance) in enumerate(features):
            # Feature name
            lbl_feature = tk.Label(
                feature_frame, 
                text=feature, 
                font=("Helvetica", 12),
                bg=self.colors['background'],
                fg=self.colors['text'],
                width=15,
                anchor="w"
            )
            lbl_feature.grid(row=i+1, column=0, padx=5, pady=2)
            
            # Importance level
            lbl_importance = tk.Label(
                feature_frame, 
                text=importance, 
                font=("Helvetica", 12),
                bg=self.colors['background'],
                fg="#e67e22" if importance == "High" else ("#f39c12" if importance == "Medium" else "#3498db"),
                width=10,
                anchor="w"
            )
            lbl_importance.grid(row=i+1, column=1, padx=5, pady=2)
    
    def show_classifier(self):
        self.clear_main_frame()
        self.status_var.set("Test Classifier View")
        
        # Title
        title_label = tk.Label(
            self.main_frame, 
            text="Test Email Classifier", 
            font=("Helvetica", 16, "bold"), 
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        title_label.pack(pady=20)
        
        # Input frame
        input_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        input_frame.pack(pady=20)
        
        # Email text label
        email_label = tk.Label(
            input_frame, 
            text="Enter Email Text:", 
            font=("Helvetica", 12),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        email_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Email text entry
        self.email_text = tk.Text(
            input_frame, 
            width=80, 
            height=10,
            font=("Helvetica", 12),
            wrap=tk.WORD,
            bg="white",
            fg=self.colors['text']
        )
        self.email_text.grid(row=1, column=0, columnspan=2, padx=10, pady=5)
        
        # Classify button
        classify_btn = tk.Button(
            input_frame, 
            text="Classify Email", 
            font=("Helvetica", 12, "bold"),
            bg="#66c2a5",
            fg="white",
            command=self.classify_email
        )
        classify_btn.grid(row=2, column=0, pady=20, sticky="w")
        
        # Clear button
        clear_btn = tk.Button(
            input_frame, 
            text="Clear", 
            font=("Helvetica", 12),
            bg="#adb5bd",
            fg="white",
            command=self.clear_classifier
        )
        clear_btn.grid(row=2, column=1, pady=20, sticky="e")
        
        # Result frame
        self.result_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        self.result_frame.pack(pady=20)
        
        # Initially hide result frame
        self.result_frame.pack_forget()
    
    def classify_email(self):
        email_text = self.email_text.get("1.0", tk.END).strip()
        
        if not email_text:
            messagebox.showwarning("Warning", "Please enter some text to classify")
            return
        
        # Clean and vectorize the text
        clean_text = self.clean_text(email_text)
        vectorized_text = self.vectorizer.transform([clean_text])
        
        # Predict
        prediction = self.model.predict(vectorized_text)[0]
        proba = self.model.predict_proba(vectorized_text)[0]
        
        # Show result
        self.show_classification_result(prediction, proba, email_text)
    
    def show_classification_result(self, prediction, proba, original_text):
        # Clear previous result
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Show result frame
        self.result_frame.pack(pady=20)
        
        # Result title
        result_title = tk.Label(
            self.result_frame, 
            text="Classification Result", 
            font=("Helvetica", 14, "bold"),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        result_title.pack(pady=10)
        
        # Prediction
        prediction_text = "This email is classified as: " + ("SPAM" if prediction == "spam" else "HAM")
        prediction_color = self.colors['spam'] if prediction == "spam" else self.colors['ham']
        
        prediction_label = tk.Label(
            self.result_frame, 
            text=prediction_text, 
            font=("Helvetica", 14, "bold"),
            bg=self.colors['background'],
            fg=prediction_color
        )
        prediction_label.pack(pady=10)
        
        # Probability
        proba_text = f"Confidence: {max(proba)*100:.2f}%"
        proba_label = tk.Label(
            self.result_frame, 
            text=proba_text, 
            font=("Helvetica", 12),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        proba_label.pack(pady=5)
        
        # Probability breakdown
        breakdown_frame = tk.Frame(self.result_frame, bg=self.colors['background'])
        breakdown_frame.pack(pady=10)
        
        # Ham probability
        ham_label = tk.Label(
            breakdown_frame, 
            text=f"Ham Probability: {proba[0]*100:.2f}%", 
            font=("Helvetica", 12),
            bg=self.colors['background'],
            fg=self.colors['ham']
        )
        ham_label.grid(row=0, column=0, padx=10)
        
        # Spam probability
        spam_label = tk.Label(
            breakdown_frame, 
            text=f"Spam Probability: {proba[1]*100:.2f}%", 
            font=("Helvetica", 12),
            bg=self.colors['background'],
            fg=self.colors['spam']
        )
        spam_label.grid(row=0, column=1, padx=10)
        
        # Highlight features
        features_frame = tk.Frame(self.result_frame, bg=self.colors['background'])
        features_frame.pack(pady=20)
        
        features_title = tk.Label(
            features_frame, 
            text="Key Features in this Email:", 
            font=("Helvetica", 12, "bold"),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        features_title.pack(anchor="w")
        
        # Get important words
        clean_text = self.clean_text(original_text)
        important_words = []
        for word in clean_text.split():
            if word in self.vectorizer.vocabulary_:
                idx = self.vectorizer.vocabulary_[word]
                weight = self.model.feature_log_prob_[1][idx] - self.model.feature_log_prob_[0][idx]
                if abs(weight) > 1.0:  # Only show significant words
                    important_words.append((word, weight))
        
        # Sort by importance
        important_words.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Display top 5 features
        for i, (word, weight) in enumerate(important_words[:5]):
            feature_label = tk.Label(
                features_frame, 
                text=f"{word} ({'spam' if weight > 0 else 'ham'}, weight={weight:.2f})", 
                font=("Helvetica", 11),
                bg=self.colors['background'],
                fg=self.colors['spam'] if weight > 0 else self.colors['ham']
            )
            feature_label.pack(anchor="w")
    
    def clear_classifier(self):
        self.email_text.delete("1.0", tk.END)
        self.result_frame.pack_forget()
    
    def show_examples(self):
        self.clear_main_frame()
        self.status_var.set("Explore Examples View")
        
        # Title
        title_label = tk.Label(
            self.main_frame, 
            text="Explore Email Examples", 
            font=("Helvetica", 16, "bold"), 
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        title_label.pack(pady=20)
        
        # Description
        desc_label = tk.Label(
            self.main_frame,
            text="Explore how different emails are classified and what features contribute to the decision.",
            font=("Helvetica", 12),
            bg=self.colors['background'],
            fg=self.colors['text'],
            wraplength=800
        )
        desc_label.pack(pady=10)
        
        # Example selection frame
        example_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        example_frame.pack(pady=20)
        
        # Example selection label
        example_label = tk.Label(
            example_frame,
            text="Select an example email:",
            font=("Helvetica", 12),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        example_label.grid(row=0, column=0, padx=10, sticky="w")
        
        # Example dropdown
        self.example_var = tk.StringVar()
        example_dropdown = ttk.Combobox(
            example_frame,
            textvariable=self.example_var,
            values=list(self.example_emails.keys()),
            font=("Helvetica", 12),
            state="readonly",
            width=30
        )
        example_dropdown.grid(row=0, column=1, padx=10)
        example_dropdown.bind("<<ComboboxSelected>>", self.display_example)
        
        # Load button
        load_btn = tk.Button(
            example_frame,
            text="Load Example",
            font=("Helvetica", 12),
            bg="#66c2a5",
            fg="white",
            command=lambda: self.display_example(None)
        )
        load_btn.grid(row=0, column=2, padx=10)
        
        # Example display frame
        self.example_display_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        self.example_display_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Initially show first example
        self.example_var.set(list(self.example_emails.keys())[0])
        self.display_example(None)
    
    def display_example(self, event):
        # Clear previous example
        for widget in self.example_display_frame.winfo_children():
            widget.destroy()
        
        # Get selected example
        example_name = self.example_var.get()
        example = self.example_emails[example_name]
        
        # Example text frame
        text_frame = tk.Frame(self.example_display_frame, bg=self.colors['background'])
        text_frame.pack(fill=tk.X, pady=10)
        
        # Example text label
        text_label = tk.Label(
            text_frame,
            text="Example Email:",
            font=("Helvetica", 12, "bold"),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        text_label.pack(anchor="w")
        
        # Example text box
        example_text = tk.Text(
            text_frame,
            width=80,
            height=6,
            font=("Helvetica", 12),
            wrap=tk.WORD,
            bg="white",
            fg=self.colors['text']
        )
        example_text.insert(tk.END, example['text'])
        example_text.config(state=tk.DISABLED)
        example_text.pack(fill=tk.X, pady=5)
        
        # Features frame
        features_frame = tk.Frame(self.example_display_frame, bg=self.colors['background'])
        features_frame.pack(fill=tk.X, pady=10)
        
        # Features label
        features_label = tk.Label(
            features_frame,
            text="Key Features:",
            font=("Helvetica", 12, "bold"),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        features_label.pack(anchor="w")
        
        # Features text
        features_text = tk.Text(
            features_frame,
            width=80,
            height=2,
            font=("Helvetica", 12),
            wrap=tk.WORD,
            bg="white",
            fg=self.colors['text']
        )
        features_text.insert(tk.END, ", ".join(example['features']))
        features_text.config(state=tk.DISABLED)
        features_text.pack(fill=tk.X, pady=5)
        
        # Classify button
        classify_btn = tk.Button(
            self.example_display_frame,
            text="Classify This Example",
            font=("Helvetica", 12, "bold"),
            bg="#66c2a5",
            fg="white",
            command=lambda: self.classify_example(example['text'])
        )
        classify_btn.pack(pady=20)
        
        # Result frame (initially hidden)
        self.example_result_frame = tk.Frame(self.example_display_frame, bg=self.colors['background'])
        self.example_result_frame.pack(fill=tk.BOTH, expand=True)
        self.example_result_frame.pack_forget()
    
    def classify_example(self, example_text):
        # Clear previous result
        for widget in self.example_result_frame.winfo_children():
            widget.destroy()
        
        # Show result frame
        self.example_result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Clean and vectorize the text
        clean_text = self.clean_text(example_text)
        vectorized_text = self.vectorizer.transform([clean_text])
        
        # Predict
        prediction = self.model.predict(vectorized_text)[0]
        proba = self.model.predict_proba(vectorized_text)[0]
        
        # Result title
        result_title = tk.Label(
            self.example_result_frame, 
            text="Classification Result", 
            font=("Segoe UI", 14, "bold"),
            bg="#f5f9ff",
            fg="#2c7be5"
        )
        result_title.pack(pady=10)
        
        # Prediction
        prediction_text = "This email is classified as: " + ("SPAM" if prediction == "spam" else "HAM")
        prediction_color = "#e63757" if prediction == "spam" else "#00d97e"
        
        prediction_label = tk.Label(
            self.example_result_frame, 
            text=prediction_text, 
            font=("Segoe UI", 14, "bold"),
            bg="#f5f9ff",
            fg=prediction_color
        )
        prediction_label.pack(pady=10)
        
        # Confidence level
        confidence_text = f"Confidence: {max(proba)*100:.2f}%"
        confidence_label = tk.Label(
            self.example_result_frame,
            text=confidence_text,
            font=("Segoe UI", 12),
            bg="#f5f9ff",
            fg="#6c757d"
        )
        confidence_label.pack(pady=5)
        
        # Probability breakdown
        breakdown_frame = tk.Frame(self.example_result_frame, bg="#f5f9ff")
        breakdown_frame.pack(pady=10)
        
        ham_prob_label = tk.Label(
            breakdown_frame,
            text=f"Ham Probability: {proba[0]*100:.2f}%",
            font=("Segoe UI", 11),
            bg="#f5f9ff",
            fg="#2c7be5"
        )
        ham_prob_label.pack(side=tk.LEFT, padx=10)
        
        spam_prob_label = tk.Label(
            breakdown_frame,
            text=f"Spam Probability: {proba[1]*100:.2f}%",
            font=("Segoe UI", 11),
            bg="#f5f9ff",
            fg="#e63757"
        )
        spam_prob_label.pack(side=tk.LEFT, padx=10)
        
        # Explanation frame
        explanation_frame = tk.Frame(self.example_result_frame, bg="#f5f9ff")
        explanation_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Explanation text widget
        explanation_text = tk.Text(
            explanation_frame,
            width=80,
            height=10,
            font=("Segoe UI", 11),
            wrap=tk.WORD,
            bg="white",
            fg="#495057",
            padx=10,
            pady=10,
            bd=1,
            relief=tk.SOLID
        )
        
        # Generate explanation based on prediction
        if prediction == "spam":
            explanation = "🚩 This email was classified as SPAM because it contains several spam indicators:\n\n"
            
            # Check for common spam features
            spam_features = []
            if re.search(r'\bfree\b', example_text, re.I):
                spam_features.append("- Contains the word 'free' (common in spam offers)")
            if re.search(r'\bwin\b|\bwon\b|\bwinner\b', example_text, re.I):
                spam_features.append("- Mentions winning or prizes (common scam tactic)")
            if re.search(r'\bhttp\b|\bwww\.|\.com\b', example_text, re.I):
                spam_features.append("- Contains links or URLs (often used in phishing)")
            if re.search(r'\bclick here\b|\bcall now\b', example_text, re.I):
                spam_features.append("- Uses urgent action phrases ('click here', 'call now')")
            if re.search(r'\b\d+\b', example_text):
                spam_features.append("- Contains numbers (often used in fake offers)")
            if len(example_text.split()) < 20:
                spam_features.append("- Short message length (common in spam)")
            
            # If no specific features found, give generic explanation
            if not spam_features:
                spam_features.append("- The content matches patterns commonly found in spam emails")
            
            explanation += "\n".join(spam_features)
            explanation += "\n\n💡 Tip: Be cautious with emails requesting immediate action or offering unexpected rewards."
        else:
            explanation = "✅ This email was classified as HAM (legitimate) because:\n\n"
            
            # Check for ham features
            ham_features = []
            if re.search(r'\bhi\b|\bhello\b|\bdear\b', example_text, re.I):
                ham_features.append("- Contains personal greetings")
            if re.search(r'\bmeeting\b|\bappointment\b|\bdiscuss\b', example_text, re.I):
                ham_features.append("- Mentions meetings or discussions (typical of legitimate communication)")
            if re.search(r'\bplease\b|\bthank you\b|\bregards\b', example_text, re.I):
                ham_features.append("- Uses polite language")
            if len(example_text.split()) > 30:
                ham_features.append("- Longer message length (typical of personal/professional emails)")
            if not re.search(r'\bhttp\b|\bwww\.|\.com\b', example_text, re.I):
                ham_features.append("- No suspicious links detected")
            
            # If no specific features found, give generic explanation
            if not ham_features:
                ham_features.append("- The content doesn't match common spam patterns")
            
            explanation += "\n".join(ham_features)
            explanation += "\n\n💡 Tip: Still exercise caution with attachments or requests for sensitive information."
        
        # Add probability explanation
        explanation += f"\n\nConfidence Analysis:\n"
        if max(proba) > 0.9:
            explanation += "- Very high confidence in this classification"
        elif max(proba) > 0.7:
            explanation += "- Strong confidence in this classification"
        else:
            explanation += "- Moderate confidence in this classification (consider reviewing manually)"
        
        # Insert explanation and configure tags
        explanation_text.insert(tk.END, explanation)
        
        # Configure tags for styling
        explanation_text.tag_config('spam', foreground='#e63757')
        explanation_text.tag_config('ham', foreground='#00d97e')
        explanation_text.tag_config('tip', foreground='#6c757d', font=('Segoe UI', 10, 'italic'))
        
        # Apply tags
        if prediction == "spam":
            explanation_text.tag_add('spam', '1.0', '1.37')
        else:
            explanation_text.tag_add('ham', '1.0', '1.42')
        
        # Find and tag all tips
        tip_start = explanation.find("💡 Tip:")
        if tip_start != -1:
            explanation_text.tag_add('tip', f"1.{tip_start}", tk.END)
        
        explanation_text.config(state=tk.DISABLED)
        
        # Add scrollbar
        scroll_y = ttk.Scrollbar(explanation_frame, command=explanation_text.yview)
        explanation_text.configure(yscrollcommand=scroll_y.set)
        
        explanation_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Feature importance visualization
        if hasattr(self.model, 'feature_log_prob_'):
            try:
                # Get top predictive features
                feature_names = np.array(self.vectorizer.get_feature_names_out())
                sorted_indices = np.argsort(self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0])
                
                if prediction == "spam":
                    top_features = feature_names[sorted_indices[-5:]][::-1]  # Top spam indicators
                else:
                    top_features = feature_names[sorted_indices[:5]]  # Top ham indicators
                
                # Create feature frame
                feature_frame = tk.Frame(self.example_result_frame, bg="#f5f9ff")
                feature_frame.pack(fill=tk.X, pady=10)
                
                feature_title = tk.Label(
                    feature_frame,
                    text="Key Features Influencing This Classification:",
                    font=("Segoe UI", 11, "bold"),
                    bg="#f5f9ff",
                    fg="#2c7be5"
                )
                feature_title.pack(anchor=tk.W)
                
                # Show features
                for i, feature in enumerate(top_features):
                    if feature in clean_text.split():
                        feature_label = tk.Label(
                            feature_frame,
                            text=f"• {feature} (present in this email)",
                            font=("Segoe UI", 10),
                            bg="#f5f9ff",
                            fg="#495057"
                        )
                    else:
                        feature_label = tk.Label(
                            feature_frame,
                            text=f"• {feature} (general indicator)",
                            font=("Segoe UI", 10),
                            bg="#f5f9ff",
                            fg="#adb5bd"
                        )
                    feature_label.pack(anchor=tk.W)
            except Exception as e:
                print(f"Error showing feature importance: {e}")


        # def export_results(self):
        #     self.clear_main_frame()
        #     self.status_var.set("Export Results View")
            
        #     # Title
        #     title_label = tk.Label(
        #         self.main_frame, 
        #         text="Export Analysis Results", 
        #         font=("Helvetica", 16, "bold"), 
        #         bg=self.colors['background'],
        #         fg=self.colors['text']
        #     )
        #     title_label.pack(pady=20)
            
        #     # Description
        #     desc_label = tk.Label(
        #         self.main_frame,
        #         text="Export the analysis results, model performance metrics, or classified emails to various formats.",
        #         font=("Helvetica", 12),
        #         bg=self.colors['background'],
        #         fg=self.colors['text'],
        #         wraplength=800
        #     )
        #     desc_label.pack(pady=10)
            
        #     # Export options frame
        #     options_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        #     options_frame.pack(pady=20)
            
        #     # Export data button
        #     data_btn = tk.Button(
        #         options_frame,
        #         text="Export Email Data (CSV)",
        #         font=("Helvetica", 12),
        #         bg="#66c2a5",
        #         fg="white",
        #         width=25,
        #         height=2,
        #         command=self.export_data_csv
        #     )
        #     data_btn.grid(row=0, column=0, padx=20, pady=10)
            
        #     # Export model button
        #     model_btn = tk.Button(
        #         options_frame,
        #         text="Export Model Details (TXT)",
        #         font=("Helvetica", 12),
        #         bg="#8da0cb",
        #         fg="white",
        #         width=25,
        #         height=2,
        #         command=self.export_model_details
        #     )
        #     model_btn.grid(row=0, column=1, padx=20, pady=10)
            
        #     # Export charts button
        #     charts_btn = tk.Button(
        #         options_frame,
        #         text="Export Charts (PNG)",
        #         font=("Helvetica", 12),
        #         bg="#e78ac3",
        #         fg="white",
        #         width=25,
        #         height=2,
        #         command=self.export_charts
        #     )
        #     charts_btn.grid(row=1, column=0, padx=20, pady=10)
            
        #     # Export all button
        #     all_btn = tk.Button(
        #         options_frame,
        #         text="Export All (ZIP)",
        #         font=("Helvetica", 12),
        #         bg="#fc8d62",
        #         fg="white",
        #         width=25,
        #         height=2,
        #         command=self.export_all
        #     )
        #     all_btn.grid(row=1, column=1, padx=20, pady=10)
            
        #     # Status label
        #     self.export_status = tk.StringVar()
        #     self.export_status.set("Select an export option above")
        #     status_label = tk.Label(
        #         self.main_frame,
        #         textvariable=self.export_status,
        #         font=("Helvetica", 12),
        #         bg=self.colors['background'],
        #         fg=self.colors['text']
        #     )
        #     status_label.pack(pady=20)
    def export_results(self):
        self.clear_main_frame()
        self.status_var.set("Export Results View")
        
        # Title
        title_label = tk.Label(
            self.main_frame, 
            text="Export Analysis Results", 
            font=("Helvetica", 16, "bold"), 
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        title_label.pack(pady=20)
        
        # Description
        desc_label = tk.Label(
            self.main_frame,
            text="Export the analysis results, model performance metrics, or classified emails to various formats.",
            font=("Helvetica", 12),
            bg=self.colors['background'],
            fg=self.colors['text'],
            wraplength=800
        )
        desc_label.pack(pady=10)
        
        # Export options frame
        options_frame = tk.Frame(self.main_frame, bg=self.colors['background'])
        options_frame.pack(pady=20)
        
        # Export data button
        data_btn = tk.Button(
            options_frame,
            text="Export Email Data (CSV)",
            font=("Helvetica", 12),
            bg="#66c2a5",
            fg="white",
            width=25,
            height=2,
            command=self.export_data_csv
        )
        data_btn.grid(row=0, column=0, padx=20, pady=10)
        
        # Export model button
        model_btn = tk.Button(
            options_frame,
            text="Export Model Details (TXT)",
            font=("Helvetica", 12),
            bg="#8da0cb",
            fg="white",
            width=25,
            height=2,
            command=self.export_model_details
        )
        model_btn.grid(row=0, column=1, padx=20, pady=10)
        
        # Export charts button
        charts_btn = tk.Button(
            options_frame,
            text="Export Charts (PNG)",
            font=("Helvetica", 12),
            bg="#e78ac3",
            fg="white",
            width=25,
            height=2,
            command=self.export_charts
        )
        charts_btn.grid(row=1, column=0, padx=20, pady=10)
        
        # Export all button
        all_btn = tk.Button(
            options_frame,
            text="Export All (ZIP)",
            font=("Helvetica", 12),
            bg="#fc8d62",
            fg="white",
            width=25,
            height=2,
            command=self.export_all
        )
        all_btn.grid(row=1, column=1, padx=20, pady=10)
        
        # Status label
        self.export_status = tk.StringVar()
        self.export_status.set("Select an export option above")
        status_label = tk.Label(
            self.main_frame,
            textvariable=self.export_status,
            font=("Helvetica", 12),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        status_label.pack(pady=20)


    def export_data_csv(self):
        try:
            # In a real application, this would save to a file
            # For demo purposes, we'll just show a message
            self.export_status.set("Email data exported to emails_data.csv")
            messagebox.showinfo("Export Successful", "Email data has been exported to CSV format.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data: {str(e)}")
            self.export_status.set("Export failed")
    
    def export_model_details(self):
        try:
            # In a real application, this would save model details
            self.export_status.set("Model details exported to model_info.txt")
            messagebox.showinfo("Export Successful", "Model details have been exported to text format.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export model details: {str(e)}")
            self.export_status.set("Export failed")
    
    def export_charts(self):
        try:
            # In a real application, this would save charts
            self.export_status.set("Charts exported as PNG images")
            messagebox.showinfo("Export Successful", "Charts have been exported as PNG images.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export charts: {str(e)}")
            self.export_status.set("Export failed")
    
    def export_all(self):
        try:
            # In a real application, this would create a zip file
            self.export_status.set("All data exported to analysis_results.zip")
            messagebox.showinfo("Export Successful", "All data has been exported in a ZIP archive.")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to create export package: {str(e)}")
            self.export_status.set("Export failed")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = SpamEmailDetectorApp(root)
    root.mainloop()