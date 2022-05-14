import os

import torch
from tqdm import tqdm

from utils.utils import get_lr

from get_criterion import get_criterion_val

def fit_one_epoch(model, train_util, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, loss_pre=0):
    total_loss_by_iters = loss_pre
    total_loss = 0
    rpn_loc_loss = 0
    rpn_cls_loss = 0
    roi_loc_loss = 0
    roi_cls_loss = 0
    
    by_iter_train = 1000
    by_iter_val = 4000
    
    val_loss = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            
            iter_total = epoch * epoch_step + iteration
            
            if iteration >= epoch_step:
                break
            images, boxes, labels = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = images.cuda()

            rpn_loc, rpn_cls, roi_loc, roi_cls, total = train_util.train_step(images, boxes, labels, 1, fp16, scaler)
            total_loss      += total.item()
            total_loss_by_iters += total.item()
            rpn_loc_loss    += rpn_loc.item()
            rpn_cls_loss    += rpn_cls.item()
            roi_loc_loss    += roi_loc.item()
            roi_cls_loss    += roi_cls.item()
            
            pbar.set_postfix(**{'total_loss'    : total_loss / (iteration + 1), 
                                'rpn_loc'       : rpn_loc_loss / (iteration + 1),  
                                'rpn_cls'       : rpn_cls_loss / (iteration + 1), 
                                'roi_loc'       : roi_loc_loss / (iteration + 1), 
                                'roi_cls'       : roi_cls_loss / (iteration + 1), 
                                'lr'            : get_lr(optimizer)})
            pbar.update(1)
            
            if iter_total % by_iter_train == 0 and iter_total != 0:
                train_avg_loss = total_loss_by_iters / by_iter_train
                loss_history.append_train_loss(train_avg_loss, iter_total)
                total_loss_by_iters = 0
            
            
            # Validation
            if iter_total % by_iter_val == 0 and iter_total != 0:
                val_loss = 0
                
                print('\n\nStart Validation')
                for iteration, batch in enumerate(gen_val):
                    if iteration >= epoch_step_val:
                        break
                    images, boxes, labels = batch[0], batch[1], batch[2]
                    with torch.no_grad():
                        if cuda:
                            images = images.cuda()

                        train_util.optimizer.zero_grad()
                        _, _, _, _, val_total = train_util.forward(images, boxes, labels, 1)
                        val_loss += val_total.item()
                        
                        if iteration % 500 == 0 and iteration != 0:
                            print('val iter:', iteration)
                            
                loss_history.append_val_loss(val_loss / epoch_step_val, iter_total)
                
                # Save the temp model
                torch.save(model.state_dict(), os.path.join(save_dir, "temp_weights.pth"))
                
                print('Start Eval')
                mAP, mIOU, mIOU_mod, acc = get_criterion_val()
                print('mAP: ', mAP)
                print('mIOU: ', mIOU)
                print('mIOU_mod: ', mIOU_mod)
                print('acc: ', acc)
                
                loss_history.append_misc(mAP, mIOU, mIOU_mod, acc, iter_total)
                
                print('Finish Validation')

    print('Finish Train')
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('Train Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
    
    
    #-----------------------------------------------#
    #   保存权值
    #-----------------------------------------------#
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
    torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))
    
    loss_history.save_data()
    
    return total_loss_by_iters