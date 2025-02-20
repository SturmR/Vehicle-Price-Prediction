
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, normaltest, pearsonr
from sklearn.inspection import permutation_importance
import traceback

class VehiclePriceAnalyzer:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = RobustScaler()
        self.model = xgb.XGBRegressor(
            n_estimators=2000, 
            learning_rate=0.005, 
            max_depth=8,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42
        )
          
    def create_advanced_features(self, df):
        df_processed = df.copy()
        luxury_brands = ['mercedes-benz', 'bmw', 'audi', 'lexus', 'porsche']
        premium_brands = ['volkswagen', 'volvo', 'subaru', 'mini'] 
        economy_brands = ['toyota', 'honda', 'mazda', 'hyundai', 'kia']
        
        df_processed['brand_category'] = df_processed['Brand'].str.lower().apply(
            lambda x: 'luxury' if x in luxury_brands else
                    'premium' if x in premium_brands else
                    'economy' if x in economy_brands else 'standard'
        )
    
        df_processed['base_depreciation'] = 1 / (1 + 0.2 * df_processed['Age'])
        
        depreciation_multipliers = {
            'luxury': 0.85,    
            'premium': 0.9,   
            'economy': 0.95,   
            'standard': 0.92 
        }
        
        df_processed['adjusted_depreciation'] = df_processed.apply(
            lambda row: row['base_depreciation'] * depreciation_multipliers[row['brand_category']], 
            axis=1
        )
        
        df_processed['model_frequency'] = df_processed.groupby('Model')['Model'].transform('count')
        df_processed['brand_frequency'] = df_processed.groupby('Brand')['Brand'].transform('count')
        
        df_processed['model_rarity'] = 1 / np.log1p(df_processed['model_frequency'])
        df_processed['brand_popularity'] = df_processed['brand_frequency'] / len(df_processed)
        
        df_processed['depreciation_score'] = (
            df_processed['adjusted_depreciation'] * 
            (1 + df_processed['model_rarity']) * 
            (1 + df_processed['brand_popularity'])
        )
        
        mileage_impact_multipliers = {
            'luxury': 1.2,    
            'premium': 1.1,   
            'economy': 0.9,    
            'standard': 1.0    
        }
        
        df_processed['mileage_impact'] = df_processed.apply(
            lambda row: (row['KmPerYear'] / 20000) * mileage_impact_multipliers[row['brand_category']],
            axis=1
        )
        
        df_processed['market_value_indicator'] = (
            df_processed['depreciation_score'] * 
            (1 - df_processed['mileage_impact']) * 
            (1 + df_processed['model_rarity'])
        )
        
        return df_processed


    def extract_title_features(self, title):
        """Extracts premium features from vehicle title"""
        title = str(title).lower()
        premium_keywords = ['premium', 'luxury', 'elite', 'ultimate', 'exclusive', 
                        'limited', 'signature', 'prestige', 'platinum']
        
        return int(any(keyword in title for keyword in premium_keywords))
        
    def preprocess_data(self, df, is_advanced):
        print("\nStarting enhanced data preprocessing...")
        df_processed = df.copy()
        
        df_processed['CylindersinEngine'] = df_processed['CylindersinEngine'].astype(str).str.extract('(\d+)').astype(float)
        
        df_processed['Price'] = pd.to_numeric(df_processed['Price'].astype(str).str.replace('[$,]', ''), errors='coerce')
        df_processed['Price_Log'] = np.log1p(df_processed['Price'])
        
        price_z_scores = np.abs((df_processed['Price'] - df_processed['Price'].mean()) / df_processed['Price'].std())
        df_processed = df_processed[price_z_scores < 3]

        model_avg_prices = df_processed.groupby('Model')['Price'].transform('mean')
    
        df_processed['market_segment_score'] = pd.qcut(
            model_avg_prices,
            q=5,
            labels=[1, 2, 3, 4, 5]  # 1 = budget, 5 = luxury
        ).astype(float)
        

        df_processed['segment_competition'] = df_processed.groupby('market_segment_score')['Model'].transform('count')
        
        df_processed['market_position'] = df_processed['market_segment_score'] * (1 / np.log1p(df_processed['segment_competition']))
        
        df_processed['Age'] = 2024 - df_processed['Year']
        df_processed['Age_Squared'] = df_processed['Age'] ** 2
        
        df_processed['Kilometres'] = pd.to_numeric(df_processed['Kilometres'].astype(str).str.replace(',', ''), errors='coerce')
        df_processed['KmPerYear'] = df_processed['Kilometres'] / df_processed['Age'].replace(0, 1)
        df_processed['KmPerYear_Log'] = np.log1p(df_processed['KmPerYear'])
        
        df_processed['IsNew'] = (df_processed['UsedOrNew'].str.upper() == 'NEW').astype(int)
        df_processed['IsDemo'] = (df_processed['UsedOrNew'].str.upper() == 'DEMO').astype(int)
        df_processed['IsUsed'] = (~df_processed['UsedOrNew'].str.upper().isin(['NEW', 'DEMO'])).astype(int)
        
        df_processed['EngineSize'] = df_processed['Engine'].str.extract(r'(\d+\.?\d*)\s*L').astype(float)
        
        df_processed['FuelConsumption_Numeric'] = df_processed['FuelConsumption'].str.extract(r'(\d+\.?\d*)').astype(float)
        
        df_processed['PriceSegment'] = pd.qcut(df_processed['Price'], q=10, labels=range(1,11))
        
        df_processed['Premium_Brand'] = df_processed['Brand'].isin(['Mercedes-Benz', 'BMW', 'Audi', 'Lexus', 'Porsche']).astype(int)
        df_processed['Is_SUV'] = df_processed['BodyType'].str.contains('SUV', na=False).astype(int)
        
        df_processed['Doors'] = df_processed['Doors'].astype(str).str.extract('(\d+)').astype(float)
        df_processed['Seats'] = df_processed['Seats'].astype(str).str.extract('(\d+)').astype(float)

        if is_advanced:
            df_processed = self.create_advanced_features(df_processed)
            
            features = [
            'Brand', 'Age', 'Age_Squared', 'BodyType', 'Transmission', 'DriveType',
            'KmPerYear_Log', 'CylindersinEngine', 'Doors', 'Seats',
            'IsNew', 'IsDemo', 'IsUsed', 'Premium_Brand', 'Is_SUV',
            'EngineSize', 'FuelConsumption_Numeric', 'market_position' ,
            'depreciation_score', 'market_value_indicator','model_rarity','brand_popularity'
        ]
            
        else:
            features = [
            'Brand', 'Age', 'Age_Squared', 'BodyType', 'Transmission', 'DriveType',
            'KmPerYear_Log', 'CylindersinEngine', 'Doors', 'Seats',
            'IsNew', 'IsDemo', 'IsUsed', 'Premium_Brand', 'Is_SUV',
            'EngineSize', 'FuelConsumption_Numeric'
            ]

        
        categorical_features = ['Brand', 'BodyType', 'Transmission', 'DriveType']
        for feature in categorical_features:
            if feature in features:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                df_processed[feature] = df_processed[feature].fillna('Unknown')
                df_processed[feature] = self.label_encoders[feature].fit_transform(df_processed[feature].astype(str))
        
        numeric_features = [f for f in features if df_processed[f].dtype in ['int64', 'float64']]
        for feature in numeric_features:
            if feature in df_processed.columns:
                if df_processed[feature].isnull().any():
                    non_null_values = df_processed[feature].dropna()
                    if len(non_null_values) > 0:
                        skewness = skew(non_null_values)
                        if abs(skewness) > 1:
                            df_processed[feature] = df_processed[feature].fillna(df_processed[feature].median())
                        else:
                            df_processed[feature] = df_processed[feature].fillna(df_processed[feature].mean())
                    else:
                        df_processed[feature] = df_processed[feature].fillna(0)
        
        X = df_processed[features]
        y = df_processed['Price_Log'] 
        
        print(f"Preprocessed data shape: X: {X.shape}, y: {y.shape}")
        return X, y
    
    
    
    def hyperparameter_tune(self, X_train, y_train):
        param_dist = {
            'n_estimators': [1000, 1500, 2000, 2500],
            'learning_rate': [0.01, 0.005, 0.001],
            'max_depth': [6, 7, 8, 9],
            'min_child_weight': [1, 2, 3, 4],
            'subsample': [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'reg_alpha': [0.1, 0.5, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0]
        }
        
        random_search = RandomizedSearchCV(
            self.model, param_distributions=param_dist,
            n_iter=20, cv=5, scoring='neg_mean_squared_error',
            random_state=42, n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        self.model = random_search.best_estimator_
        print(f"\nBest parameters: {random_search.best_params_}")

    
    def train(self, X_train, y_train, tune_hyperparameters=True):
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if tune_hyperparameters:
            self.hyperparameter_tune(X_train_scaled, y_train)
        
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, 
                                  cv=5, scoring='r2')
        print(f"\nCross-validation R² scores: {cv_scores}")
        print(f"Mean CV R² score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.model.fit(X_train_scaled, y_train)
    
    def evaluate(self, X_test, y_test_log):
        """Evaluate the model's performance with comprehensive visualizations"""
        y_pred_log = self.model.predict(self.scaler.transform(X_test))
        y_test = np.expm1(y_test_log)
        y_pred = np.expm1(y_pred_log)
        
        metrics = {
            'R2 Score': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        # Create main performance visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted Plot
        axes[0,0].scatter(y_test, y_pred, alpha=0.5, color='blue', s=20)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0,0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Price ($)')
        axes[0,0].set_ylabel('Predicted Price ($)')
        axes[0,0].set_title('Actual vs Predicted Vehicle Prices')
        axes[0,0].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        axes[0,0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Residuals Plot
        residuals = y_test - y_pred
        axes[0,1].scatter(y_pred, residuals, alpha=0.5, color='green', s=20)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predicted Price ($)')
        axes[0,1].set_ylabel('Residuals ($)')
        axes[0,1].set_title('Residuals vs Predicted Price')
        axes[0,1].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        axes[0,1].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
        
        # Feature Importance Plot
        importance_df = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        axes[1,0].barh(range(len(importance_df)), importance_df['importance'], color='skyblue')
        axes[1,0].set_yticks(range(len(importance_df)))
        axes[1,0].set_yticklabels(importance_df['feature'])
        axes[1,0].set_xlabel('Feature Importance')
        axes[1,0].set_title('Feature Importance Rankings')
        
        # Error Distribution Plot
        percentage_errors = (residuals / y_test) * 100
        percentage_errors = percentage_errors[np.abs(percentage_errors) < np.percentile(np.abs(percentage_errors), 95)]
        axes[1,1].hist(percentage_errors, bins=50, color='purple', alpha=0.7)
        axes[1,1].set_xlabel('Percentage Error (%)')
        axes[1,1].set_title('Error Distribution (Percentage)')
        
        plt.tight_layout()
        plt.show()
        
        # Print performance metrics
        print("\nDetailed Model Performance Metrics:")
        print("=" * 50)
        for metric, value in metrics.items():
            if metric in ['R2 Score', 'MAPE']:
                print(f"{metric}: {value:.2f}%")
            else:
                print(f"{metric}: ${value:,.2f}")
        
        return metrics, importance_df

    def predict(self, X):
        """
        Make predictions on new data
        """
        X_scaled = self.scaler.transform(X)
        predictions_log = self.model.predict(X_scaled)
        return np.expm1(predictions_log)  
    
    def analyze_feature_impact(self, X, y):
        """
        Analyze the impact of new features through comparative modeling with separate scalers
        """
        # Split features into base and advanced sets
        base_features = [
            'Brand', 'Age', 'Age_Squared', 'BodyType', 'Transmission', 'DriveType',
            'KmPerYear_Log', 'CylindersinEngine', 'Doors', 'Seats',
            'IsNew', 'IsDemo', 'IsUsed', 'Premium_Brand', 'Is_SUV',
            'EngineSize', 'FuelConsumption_Numeric'
        ]
        
        advanced_features = [
            'depreciation_score', 'market_value_indicator', 
            'model_rarity', 'brand_popularity', 'market_position'
        ]
        
        base_scaler = RobustScaler()
        enhanced_scaler = RobustScaler()

        X_base = X[base_features]
        X_full = X[base_features + advanced_features]
        
        X_train_base, X_test_base, y_train, y_test = train_test_split(
            X_base, y, test_size=0.2, random_state=42
        )
        X_train_full, X_test_full, _, _ = train_test_split(
            X_full, y, test_size=0.2, random_state=42
        )
        
        # Scale the data
        X_train_base_scaled = base_scaler.fit_transform(X_train_base)
        X_test_base_scaled = base_scaler.transform(X_test_base)
        
        X_train_full_scaled = enhanced_scaler.fit_transform(X_train_full)
        X_test_full_scaled = enhanced_scaler.transform(X_test_full)
        
        base_model = xgb.XGBRegressor(**self.model.get_params())
        base_model.fit(X_train_base_scaled, y_train)
        
        enhanced_model = xgb.XGBRegressor(**self.model.get_params())
        enhanced_model.fit(X_train_full_scaled, y_train)
        
        y_pred_base = np.expm1(base_model.predict(X_test_base_scaled))
        y_pred_enhanced = np.expm1(enhanced_model.predict(X_test_full_scaled))
        y_test_actual = np.expm1(y_test)
        
        metrics_comparison = pd.DataFrame({
            'Base Model': [
                r2_score(y_test_actual, y_pred_base),
                np.sqrt(mean_squared_error(y_test_actual, y_pred_base)),
                mean_absolute_error(y_test_actual, y_pred_base),
                np.mean(np.abs((y_test_actual - y_pred_base) / y_test_actual)) * 100
            ],
            'Enhanced Model': [
                r2_score(y_test_actual, y_pred_enhanced),
                np.sqrt(mean_squared_error(y_test_actual, y_pred_enhanced)),
                mean_absolute_error(y_test_actual, y_pred_enhanced),
                np.mean(np.abs((y_test_actual - y_pred_enhanced) / y_test_actual)) * 100
            ]
        }, index=['R² Score', 'RMSE', 'MAE', 'MAPE'])
        
        importance_comparison = pd.DataFrame({
            'feature': X_full.columns,
            'importance': enhanced_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        metrics_comparison.plot(kind='bar', ax=ax1)
        ax1.set_title('Performance Metrics Comparison')
        ax1.set_ylabel('Score')
        ax1.legend(['Base Model', 'Model'])
        
        importance_comparison.head(15).plot(
            kind='barh', x='feature', y='importance', ax=ax2
        )
        ax2.set_title('Top 15 Most Important Features')
        
        plt.tight_layout()
        plt.show()
        
        print("\nMetrics Comparison:")
        print(metrics_comparison)
        print("\nTop 10 Most Important Features:")
        print(importance_comparison.head(10))
        
        return metrics_comparison, importance_comparison
    
def plot_train_test_errors(model, X_train, y_train, X_test, y_test):
    """
    Plots train vs. test error over boosting iterations.
    """
    # Calculate predictions for train and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Compute errors
    train_error = mean_squared_error(y_train, y_train_pred, squared=False)  # RMSE
    test_error = mean_squared_error(y_test, y_test_pred, squared=False)    # RMSE

    print(f"Train RMSE: {train_error:.2f}, Test RMSE: {test_error:.2f}")
    
    # Plot error values
    plt.figure(figsize=(10, 6))
    plt.bar(['Train RMSE', 'Test RMSE'], [train_error, test_error], color=['blue', 'orange'])
    plt.ylabel('RMSE')
    plt.title('Train vs. Test Error')
    plt.show()

def plot_feature_importances(model, feature_names):
    """
    Plots feature importances based on the model.
    """
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importances)), importances[sorted_idx], align="center")
    plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Feature Importances")
    plt.show()

def plot_predictions_distribution(y_test, y_pred):
    """
    Plots the distribution of actual vs. predicted values.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(np.expm1(y_test), label="Actual", kde=True, color="blue", bins=30)
    sns.histplot(np.expm1(y_pred), label="Predicted", kde=True, color="orange", bins=30)
    plt.legend()
    plt.title("Actual vs. Predicted Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Density")
    plt.show()

def plot_residuals(y_test, y_pred):
    """
    Plots residuals (errors) for the test set.
    """
    residuals = np.expm1(y_test) - np.expm1(y_pred)
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color="red", bins=30)
    plt.title("Residuals Distribution")
    plt.xlabel("Residuals (Actual - Predicted)")
    plt.ylabel("Density")
    plt.show()

def plot_predictions_scatter(y_test, y_pred):
    """
    Plots predicted vs actual prices for the test set.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(np.expm1(y_test), np.expm1(y_pred), alpha=0.5, color='blue')
    plt.plot([min(np.expm1(y_test)), max(np.expm1(y_test))],
             [min(np.expm1(y_test)), max(np.expm1(y_test))],
             color='red', linestyle='--')
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.show()


def advanced_features_models():
    try:
        print("Starting comprehensive vehicle price analysis...")
        try:
            df = pd.read_csv('vehicle_prices.csv')
            print(f"Data loaded successfully. Shape: {df.shape}")
        except FileNotFoundError:
            print("Error: Could not find the vehicle_prices.csv file")
            return None, None
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty")
            return None, None
            
        analyzer = VehiclePriceAnalyzer()
        X, y = analyzer.preprocess_data(df,True)

        
        if len(X) == 0:
            print("Error: No valid records after preprocessing")
            return None, None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("\nTraining Advanced features model...")
        analyzer.train(X_train, y_train)
        
        print("\nEvaluating Advanced features model...")
        metrics, feature_importance = analyzer.evaluate(X_test, y_test)
        report = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'data_shape': df.shape
        }


        return analyzer, report
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        traceback.print_exc()
        return None, None

def model():
    try:
        print("Starting comprehensive vehicle price analysis...")
        try:
            df = pd.read_csv('vehicle_prices.csv')
            print(f"Data loaded successfully. Shape: {df.shape}")
        except FileNotFoundError:
            print("Error: Could not find the vehicle_prices.csv file")
            return None, None
        except pd.errors.EmptyDataError:
            print("Error: The CSV file is empty")
            return None, None
            
        analyzer = VehiclePriceAnalyzer()
        X, y = analyzer.preprocess_data(df, False)

        
        if len(X) == 0:
            print("Error: No valid records after preprocessing")
            return None, None
            
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print("\nTraining features model...")
        analyzer.train(X_train, y_train)
        
        print("\nEvaluating features model...")
        metrics, feature_importance = analyzer.evaluate(X_test, y_test)
        report = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'data_shape': df.shape
        }


        return analyzer, report
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    analyzer2 , report2 = advanced_features_models()
    if analyzer2 is not None and report2 is not None:
        print("\nAnalysis completed successfully")
    else:
        print("\nAnalysis failed to complete")