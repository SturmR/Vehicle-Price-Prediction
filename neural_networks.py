import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
import traceback
import random
from sklearn.model_selection import KFold

class RandomHyperparameterTuner:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def generate_random_params(self):
        """Generate random hyperparameters from defined distributions"""
        lr = np.exp(random.uniform(np.log(1e-5), np.log(1e-2)))
        
        batch_sizes = [16, 32, 48, 64, 96, 128, 256]
        batch_size = random.choice(batch_sizes)
        
        dropout_base = random.uniform(0.1, 0.5)
        dropout_rates = (
            dropout_base,
            max(0.1, dropout_base - random.uniform(0.1, 0.2)),
            max(0.1, dropout_base - random.uniform(0.2, 0.3))
        )
        
        max_units = random.choice([128, 256, 512, 1024])
        hidden_units = (
            max_units,
            max_units // 2,
            max_units // 4
        )
        
        optimizer_type = random.choice(['Adam', 'AdamW', 'RMSprop'])
        weight_decay = random.choice([0, 1e-5, 1e-4, 1e-3])
        
        return {
            'learning_rate': lr,
            'batch_size': batch_size,
            'dropout_rates': dropout_rates,
            'hidden_units': hidden_units,
            'optimizer_type': optimizer_type,
            'weight_decay': weight_decay
        }
    
    def random_search(self, n_trials=20):
        """Perform random search for hyperparameter optimization"""
        best_score = float('inf')
        best_params = None
        results = []
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        print(f"Performing {n_trials} random trials...")
        
        for trial in range(n_trials):
            params = self.generate_random_params()
            fold_scores = []
            
            print(f"\nTrial {trial + 1}/{n_trials}")
            print("Parameters:")
            for param, value in params.items():
                print(f"{param}: {value}")
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                
                model = VehiclePriceAnalyzerNN(
                    learning_rate=params['learning_rate'],
                    batch_size=params['batch_size'],
                    epochs=50,  
                    dropout_rates=params['dropout_rates'],
                    hidden_units=params['hidden_units'],
                    optimizer_type=params['optimizer_type'],
                    weight_decay=params['weight_decay']
                )
                
                try:
                    model.train(X_train, y_train, X_val, y_val)
                    val_loss, _ = model.evaluate(X_val, y_val)
                    fold_scores.append(val_loss)
                except Exception as e:
                    print(f"Error in fold {fold}: {str(e)}")
                    fold_scores.append(float('inf'))
            
            # Calculate average score across folds
            avg_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            
            results.append({
                'params': params,
                'score': avg_score,
                'std': std_score
            })
            
            print(f"Average validation loss: {avg_score:.4f} (±{std_score:.4f})")
            
            # Update best parameters if better score found
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
                print("New best score found!")
            
            print("-" * 50)
        
        return best_params, results


class VehicleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class VehiclePriceNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units=(256, 128, 64, 32), dropout_rates=(0.3, 0.2, 0.1)):
        super(VehiclePriceNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, (units, dropout) in enumerate(zip(hidden_units[:-1], dropout_rates)):
            layers.extend([
                nn.Linear(prev_dim, units),
                nn.ReLU(),
                nn.BatchNorm1d(units),
                nn.Dropout(dropout)
            ])
            prev_dim = units
        
        layers.extend([
            nn.Linear(prev_dim, hidden_units[-1]),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_units[-1])
        ])
        
        layers.append(nn.Linear(hidden_units[-1], 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class VehiclePriceAnalyzerNN:
    def __init__(self, learning_rate=0.001, batch_size=64, epochs=100, 
                 dropout_rates=(0.3, 0.2, 0.1), hidden_units=(256, 128, 64, 32),
                 optimizer_type='Adam', weight_decay=0):
        self.label_encoders = {}
        self.scaler = RobustScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rates = dropout_rates
        self.hidden_units = hidden_units
        self.optimizer_type = optimizer_type
        self.weight_decay = weight_decay
        self.model = None
        self.final_y_true = None
        print(f"Using device: {self.device}")

        
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

        
     
    def calculate_metrics(self, y_true, y_pred, is_final=False):
        """
        Calculate regression metrics for model evaluation
        """
        y_true_exp = np.expm1(y_true)
        y_pred_exp = np.expm1(y_pred)
        
        metrics = {
            'R2 Score': r2_score(y_true_exp, y_pred_exp),
            'RMSE': np.sqrt(mean_squared_error(y_true_exp, y_pred_exp)),
            'MAE': mean_absolute_error(y_true_exp, y_pred_exp),
            'MAPE': np.mean(np.abs((y_true_exp - y_pred_exp) / y_true_exp)) * 100
        }
        
        if is_final:
            print("\nModel Performance Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
                
        return metrics

    def evaluate(self, X_test, y_test, criterion=None, is_final=False):
        """
        Evaluate the model's performance
        """
        if criterion is None:
            criterion = nn.MSELoss()
            
        X_test_scaled = self.scaler.transform(X_test)
        test_dataset = VehicleDataset(X_test_scaled, y_test.values)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        total_loss = 0
        predictions = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                total_loss += criterion(outputs, batch_y).item()
                predictions.extend(outputs.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        predictions = np.array(predictions).reshape(-1)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test.values, predictions, is_final=is_final)
        
        # Only plot if it's the final evaluation
        if is_final:
            self.plot_predictions(y_test.values, predictions)
        
        return avg_loss, predictions

    def plot_training_history(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.show()


    def plot_predictions(self, y_true, y_pred):
        y_true_exp = np.expm1(y_true)
        y_pred_exp = np.expm1(y_pred)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        axes[0,0].scatter(y_true_exp, y_pred_exp, alpha=0.5)
        axes[0,0].plot([y_true_exp.min(), y_true_exp.max()], 
                      [y_true_exp.min(), y_true_exp.max()], 
                      'r--')
        axes[0,0].set_xlabel('Actual Price')
        axes[0,0].set_ylabel('Predicted Price')
        axes[0,0].set_title('Actual vs Predicted Prices')
        
        residuals = y_pred_exp - y_true_exp
        axes[0,1].scatter(y_pred_exp, residuals, alpha=0.5)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('Predicted Price')
        axes[0,1].set_ylabel('Residuals')
        axes[0,1].set_title('Residuals vs Predicted Price')
        
        axes[1,0].hist(residuals, bins=50, alpha=0.75)
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Error Distribution')
        
        percentage_errors = (residuals / y_true_exp) * 100
        percentage_errors = percentage_errors[np.abs(percentage_errors) < np.percentile(np.abs(percentage_errors), 95)]
        axes[1,1].hist(percentage_errors, bins=50, alpha=0.75)
        axes[1,1].set_xlabel('Percentage Error (%)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Percentage Error Distribution')
        
        plt.tight_layout()
        plt.show()

    def train(self, X_train, y_train, X_val=None, y_val=None):
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_dataset = VehicleDataset(X_val_scaled, y_val.values)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        train_dataset = VehicleDataset(X_train_scaled, y_train.values)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = VehiclePriceNetwork(
            input_dim=X_train.shape[1],
            hidden_units=self.hidden_units,
            dropout_rates=self.dropout_rates
        ).to(self.device)
        
        criterion = nn.MSELoss()
        initial_lr = self.learning_rate / 10
        if self.optimizer_type == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=initial_lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(self.model.parameters(), lr=initial_lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'RMSprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=initial_lr, weight_decay=self.weight_decay)
        
        # Warmup over first 5 epochs, then use original learning rate
        def adjust_learning_rate(epoch):
            if epoch < 5:
                lr = initial_lr + (self.learning_rate - initial_lr) * (epoch / 5)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        patience = 10 
        patience_counter = 0
        
        for epoch in range(self.epochs):
            adjust_learning_rate(epoch) 
            self.model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            if X_val is not None:
                val_loss, _ = self.evaluate(X_val, y_val, criterion)
                val_losses.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                scheduler.step(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("\nLoaded best model from validation")
        
        self.plot_training_history(train_losses, val_losses)

    def predict(self, X):
        """
        Make predictions on new data
        """
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return np.expm1(predictions.cpu().numpy())

def train_with_best_params():
    try:
        print("Starting training with best parameters...")
        
        # Load data
        df = pd.read_csv('vehicle_prices.csv')
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Initialize analyzer with best parameters
        best_params = {
            'learning_rate': 0.003382959517103582,
            'batch_size': 64,
            'epochs': 75,
            'dropout_rates': (0.10225583049619105, 0.1, 0.1),
            'hidden_units': (128, 64, 32),
            'optimizer_type': 'RMSprop',
            'weight_decay': 0
        }
        
        analyzer = VehiclePriceAnalyzerNN(**best_params)
        
        # Preprocess data
        X, y = analyzer.preprocess_data(df, False)
        
        if len(X) == 0:
            print("Error: No valid records after preprocessing")
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print("\nTraining neural network with best parameters...")
        analyzer.train(X_train, y_train, X_val, y_val)
        
        print("\nFinal model evaluation:")
        test_loss, predictions = analyzer.evaluate(X_test, y_test, is_final=True)
        print(f"\nFinal test loss: {test_loss:.4f}")
        
        return analyzer, X_test, y_test, predictions
        
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        traceback.print_exc()
        return None, None, None, None

if __name__ == "__main__":
    model, X_test, y_test, predictions = train_with_best_params()
    if model is not None:
        print("\nTraining completed successfully")
    else:
        print("\nTraining failed to complete")
