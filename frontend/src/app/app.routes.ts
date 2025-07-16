import { Routes } from '@angular/router';
import { ValidacionComponent } from './validacion/validacion.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { LoginComponent } from './login/login.component';
import { AuthGuard } from '../auth.guard'; ;

export const routes: Routes = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },  
  { path: 'login', component: LoginComponent },
  { path: 'dashboard', component: DashboardComponent, canActivate: [AuthGuard] },
  { path: 'validacion', component: ValidacionComponent },
  { path: '**', redirectTo: '' },  
];
