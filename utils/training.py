import torch
import math


def train_text_model(num_epochs=1, model=None, optimizer=None,
                     train_loader=None, val_loader=None,
                     criterion=None, save_path=None, clip=1.0):
    
    best_val_loss = float(math.inf)
    for epoch in range(num_epochs):
        avg_train_loss = 0
        tot_train_loss = 0
        tot_train_samples = 0
        
        model.train()
        for i, batch in enumerate(train_loader):
            batch = [el.cuda() for el in batch]
            tokens, attention_mask, nonText, target = batch
            optimizer.zero_grad()
            output = model(tokens, attention_mask).squeeze(1)
            train_loss = criterion(output, target)
            tot_train_loss += train_loss.item()
            tot_train_samples += tokens.shape[0]
            
            train_loss.backward()  # get gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # clip gradients
            optimizer.step()  # update weights
            
        avg_train_loss = tot_train_loss / tot_train_samples
        
        # Val loss
        avg_val_loss = 0
        tot_val_loss = 0
        tot_val_samples = 0
        
        model.eval()
        for i, batch in enumerate(val_loader):
            batch = [el.cuda() for el in batch]
            tokens, attention_mask, nonText, target = batch
            optimizer.zero_grad()
            output = model(tokens, attention_mask).squeeze(1)  # get outputs
            val_loss = criterion(output, target)
            tot_val_loss += val_loss.item()
            tot_val_samples += tokens.shape[0]
            
        avg_val_loss = tot_val_loss / tot_val_samples
        
        if (avg_val_loss < best_val_loss):
            torch.save(model, save_path)
            print (f'Epoch {epoch}, val loss: {best_val_loss:.8f} -> {avg_val_loss:.8f}, train loss: {avg_train_loss:.8f}')
            best_val_loss = avg_val_loss
        else:
            print (f'Epoch {epoch}, val loss: {avg_val_loss:.8f}, train loss: {avg_train_loss:.8f}')



def train_text_meta_model(num_epochs=1, model=None, optimizer=None,
                          train_loader=None, val_loader=None,
                          criterion=None, save_path=None, clip=1.0):
    
    best_val_loss = float(math.inf)
    for epoch in range(num_epochs):
        avg_train_loss = 0
        tot_train_loss = 0
        tot_train_samples = 0
        
        model.train()
        for i, batch in enumerate(train_loader):
            batch = [el.cuda() for el in batch]
            tokens, attention_mask, nonText, target = batch
            optimizer.zero_grad()
            output = model(tokens, attention_mask, nonText).squeeze(1)
            train_loss = criterion(output, target)
            tot_train_loss += train_loss.item()
            tot_train_samples += tokens.shape[0]
            
            train_loss.backward()  # get gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # clip gradients
            optimizer.step()  # update weights
            
        avg_train_loss = tot_train_loss / tot_train_samples
        
        # Val loss
        avg_val_loss = 0
        tot_val_loss = 0
        tot_val_samples = 0
        
        model.eval()
        for i, batch in enumerate(val_loader):
            batch = [el.cuda() for el in batch]
            tokens, attention_mask, nonText, target = batch
            optimizer.zero_grad()
            output = model(tokens, attention_mask, nonText).squeeze(1)  # get outputs
            val_loss = criterion(output, target)
            tot_val_loss += val_loss.item()
            tot_val_samples += tokens.shape[0]
            
        avg_val_loss = tot_val_loss / tot_val_samples
        
        if (avg_val_loss < best_val_loss):
            torch.save(model, save_path)
            print (f'Epoch {epoch}, val loss: {best_val_loss:.8f} -> {avg_val_loss:.8f}, train loss: {avg_train_loss:.8f}')
            best_val_loss = avg_val_loss
        else:
            print (f'Epoch {epoch}, val loss: {avg_val_loss:.8f}, train loss: {avg_train_loss:.8f}')



def train_meta_model(num_epochs=1, model=None, optimizer=None,
                     train_loader=None, val_loader=None,
                     criterion=None, save_path=None, clip=1.0):
    
    best_val_loss = float(math.inf)
    for epoch in range(num_epochs):
        avg_train_loss = 0
        tot_train_loss = 0
        tot_train_samples = 0
        
        model.train()
        for i, batch in enumerate(train_loader):
            batch = [el.cuda() for el in batch]
            tokens, attention_mask, nonText, target = batch
            optimizer.zero_grad()
            output = model(nonText).squeeze(1)
            train_loss = criterion(output, target)
            tot_train_loss += train_loss.item()
            tot_train_samples += tokens.shape[0]
            
            train_loss.backward()  # get gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # clip gradients
            optimizer.step()  # update weights
            
        avg_train_loss = tot_train_loss / tot_train_samples
        
        # Val loss
        avg_val_loss = 0
        tot_val_loss = 0
        tot_val_samples = 0
        
        model.eval()
        for i, batch in enumerate(val_loader):
            batch = [el.cuda() for el in batch]
            tokens, attention_mask, nonText, target = batch
            optimizer.zero_grad()
            output = model(nonText).squeeze(1)  # get outputs
            val_loss = criterion(output, target)
            tot_val_loss += val_loss.item()
            tot_val_samples += tokens.shape[0]
            
        avg_val_loss = tot_val_loss / tot_val_samples
        
        if (avg_val_loss < best_val_loss):
            torch.save(model, save_path)
            print (f'Epoch {epoch}, val loss: {best_val_loss:.8f} -> {avg_val_loss:.8f}, train loss: {avg_train_loss:.8f}')
            best_val_loss = avg_val_loss
        else:
            print (f'Epoch {epoch}, val loss: {avg_val_loss:.8f}, train loss: {avg_train_loss:.8f}')